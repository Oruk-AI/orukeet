"""Bounded ASR recovery: train all parameters except fitted Gabor rows."""
import json
import copy
import os
from pathlib import Path
import sys
import types
import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from omegaconf import OmegaConf
from fit import sha, SOURCE_SHA
from frozen import install, verify, materialized_state_dict
from anchor_loss import normalized_mse
from posterior import PosteriorAnchor
from layer_anchor import LayerAnchor
from continuation import load_candidate
from optimizer_groups import build_groups

ROOT=Path(os.environ.get('PARAKEET_ROOT','/home/nathanroll/parakeet-ft'))
sys.path.insert(0,str(ROOT/'Speech/examples/asr'))
from speech_to_text_finetune import get_base_model,check_vocabulary,setup_dataloaders
from nemo.collections.asr.models import ASRModel
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg


def save_materialized(model,dest):
    """Export regular NeMo keys without changing optimizer parameter objects."""
    if dest.exists():raise FileExistsError(dest)
    original=model.state_dict
    def state_dict(module,*args,**kwargs):
        return materialized_state_dict(module,original(*args,**kwargs))
    model.state_dict=types.MethodType(state_dict,model)
    try:model.save_to(str(dest))
    finally:del model.state_dict

class GuardAndExport(pl.Callback):
    def __init__(self,out,receipts,interval,preserve_bn=False,resume_keep=2):
        self.out=out;self.receipts=receipts;self.interval=interval;self.last=-1
        self.preserve_bn=preserve_bn;self.bn_state=None
        self.resume_keep=resume_keep
    def on_train_batch_start(self,trainer,model,batch,batch_idx):
        if self.preserve_bn:
            for module in model.modules():
                if isinstance(module,_BatchNorm):module.eval()
    def on_fit_start(self,trainer,model):
        verify(model,self.receipts)
        if self.preserve_bn:
            self.bn_state={n:b.detach().cpu().clone() for n,b in model.named_buffers() if n.endswith(('running_mean','running_var','num_batches_tracked'))}
        expected={id(p) for p in model.parameters()}
        optimized={id(p) for optimizer in trainer.optimizers for group in optimizer.param_groups for p in group['params']}
        if expected!=optimized:raise RuntimeError('Optimizer does not cover every non-Gabor parameter')
        (self.out/'optimizer-coverage.json').write_text(json.dumps({'status':'pass','parameter_tensors':len(expected),
            'trainable_scalar_parameters':sum(p.numel() for p in model.parameters()),'frozen_gabor_rows':12288,
            'groups':[{'tensors':len(g['params']),'initial_lr':g.get('initial_lr',g['lr'])}
                      for opt in trainer.optimizers for g in opt.param_groups]},indent=2)+'\n')
    def on_after_backward(self,trainer,model):
        if (self.out/'gradient-coverage.json').exists():return
        coverage={name:{'trainable':p.requires_grad,'has_gradient':p.grad is not None,
                       'finite':bool(torch.isfinite(p.grad).all()) if p.grad is not None else None}
                  for name,p in model.named_parameters()}
        for prefix in ['encoder.layers.0.','decoder.','joint.']:
            if not any(n.startswith(prefix) and v['has_gradient'] for n,v in coverage.items()):
                raise RuntimeError('Missing gradient in '+prefix)
        if any(not v['trainable'] or not v['has_gradient'] or not v['finite'] for v in coverage.values()):
            raise RuntimeError('Every non-Gabor parameter must have a finite gradient')
        (self.out/'gradient-coverage.json').write_text(json.dumps(coverage,indent=2)+'\n')
    def on_train_batch_end(self,trainer,model,outputs,batch,batch_idx):
        step=trainer.global_step
        if step==self.last:return
        self.last=step
        if step%10==0:
            metrics={k:float(v.detach().cpu()) for k,v in trainer.callback_metrics.items() if torch.is_tensor(v) and v.numel()==1}
            with (self.out/'metrics.jsonl').open('a') as f:f.write(json.dumps({'step':step,**metrics})+'\n')
            print('RECOVERY',step,json.dumps(metrics),flush=True)
        if step%25==0:verify(model,self.receipts)
        if step and step%self.interval==0:
            self.export(model,step)
            dest=self.out/f'resume-{step:04d}.ckpt'
            trainer.save_checkpoint(str(dest))
            # Only delete this run's obsolete optimizer checkpoints; keep two.
            for old in sorted(self.out.glob('resume-*.ckpt'))[:-self.resume_keep]:old.unlink()
    def export(self,model,step):
        verify(model,self.receipts)
        if self.bn_state is not None:
            buffers=dict(model.named_buffers())
            for name,value in self.bn_state.items():
                if not torch.equal(buffers[name].detach().cpu(),value):raise RuntimeError('BN state changed: '+name)
        dest=self.out/f'step-{step:04d}.nemo'
        if dest.exists():return
        save_materialized(model,dest)
        receipt={'step':step,'sha256':sha(dest),'frozen_rows_verified':12288,'frozen_coefficients':12288*9,'all_other_parameters_trainable':True}
        dest.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print('EXPORTED',json.dumps(receipt),flush=True)
    def on_train_end(self,trainer,model):self.export(model,trainer.global_step)

class EncoderAnchor(pl.Callback):
    """Match the original encoder on the same features; teacher is never exported."""
    def __init__(self,model,scale,kind='cosine',asr_scale=1.):
        self.teacher=copy.deepcopy(model.encoder).eval().requires_grad_(False)
        if kind not in ('cosine','normalized_mse'):raise ValueError(kind)
        if not scale>0 or not asr_scale>0:raise ValueError('Both objectives must remain active')
        self.scale=scale;self.loss=None;self.kind=kind
        def hook(module,args,kwargs,output):
            if not model.training or not torch.is_grad_enabled():return
            with torch.no_grad():target,_=self.teacher(*args,**kwargs)
            actual,lengths=output
            actual=actual.float();target=target.float()
            mask=torch.arange(actual.shape[2],device=actual.device)[None,:]<lengths[:,None]
            if kind=='cosine':
                similarity=(F.normalize(actual,dim=1)*F.normalize(target,dim=1)).sum(dim=1)
                self.loss=((1-similarity)*mask).sum()/mask.sum().clamp_min(1)
            else:
                self.loss=normalized_mse(actual,target,lengths)
        self.handle=model.encoder.register_forward_hook(hook,with_kwargs=True)
        original=model.training_step
        def training_step(module,batch,batch_idx):
            self.loss=None;result=original(batch,batch_idx)
            if self.loss is None:raise RuntimeError('Encoder reference did not run')
            loss=result['loss'] if isinstance(result,dict) else result
            total=asr_scale*loss+self.scale*self.loss
            module.log('encoder_anchor_'+kind,self.loss.detach(),on_step=True)
            module.log('recovery_total_loss',total.detach(),on_step=True)
            if isinstance(result,dict):result['loss']=total;return result
            return total
        model.training_step=types.MethodType(training_step,model)
    def on_fit_start(self,trainer,model):self.teacher.to(model.device).eval()

@hydra_runner(config_path='conf/asr_finetune',config_name='speech_to_text_finetune')
def main(cfg):
    source=Path(cfg.init_from_nemo_model)
    if sha(source)!=SOURCE_SHA:raise ValueError('Recovery must start from the exact original Orukeet checkpoint')
    out=Path(cfg.recovery_output);out.mkdir(parents=True,exist_ok=False)
    pl.seed_everything(int(cfg.seed),workers=True)
    trainer=pl.Trainer(**resolve_trainer_cfg(cfg.trainer))
    exp_manager(trainer,cfg.get('exp_manager'))
    model=check_vocabulary(get_base_model(trainer,cfg),cfg)
    if sum(float(cfg.get(key,0))>0 for key in ('posterior_anchor_scale','encoder_anchor_scale','layer_anchor_scale'))>1:
        raise ValueError('Choose one teacher objective')
    if float(cfg.get('layer_posterior_scale',0)) and not float(cfg.get('layer_anchor_scale',0)):
        raise ValueError('Shared posterior matching requires the layer teacher')
    if float(cfg.get('layer_anchor_scale',0)):
        trainer.callbacks.append(LayerAnchor(model,float(cfg.layer_anchor_scale),
            float(cfg.get('conv_anchor_scale',1.)),float(cfg.get('asr_loss_scale',1.)),
            float(cfg.get('layer_posterior_scale',0.)),int(cfg.get('posterior_points',16))))
    if float(cfg.get('posterior_anchor_scale',0)):
        if float(cfg.get('encoder_anchor_scale',0)):raise ValueError('Choose one teacher objective')
        trainer.callbacks.append(PosteriorAnchor(model,float(cfg.posterior_anchor_scale),
            float(cfg.get('asr_loss_scale',1.)),int(cfg.get('posterior_points',16))))
    if float(cfg.get('encoder_anchor_scale',0)):
        trainer.callbacks.append(EncoderAnchor(model,float(cfg.encoder_anchor_scale),
            str(cfg.get('encoder_anchor_kind','cosine')),float(cfg.get('asr_loss_scale',1.))))
    start_sha=SOURCE_SHA
    if cfg.get('recovery_start_model'):
        start_sha=load_candidate(model,cfg.recovery_start_model,cfg.recovery_start_audit)
    receipts=install(model,cfg.gabor_fits)
    if cfg.get('disable_dropout',False):
        for module in model.modules():
            if isinstance(module,torch.nn.Dropout):module.p=0.
            if isinstance(module,torch.nn.RNNBase):module.dropout=0.
    guard=GuardAndExport(out,receipts,int(cfg.export_interval),bool(cfg.get("preserve_bn_statistics",False)),int(cfg.get('resume_keep',2)))
    trainer.callbacks.append(guard)
    model=setup_dataloaders(model,cfg)
    if cfg.get('recovery_param_groups') is not None:
        OmegaConf.set_struct(model.cfg,False)
        model.cfg.optim_param_groups=cfg.recovery_param_groups
    if cfg.get('recovery_layer_lrs') is not None:
        if cfg.get('recovery_param_groups') is not None:raise ValueError('Choose one parameter grouping')
        rates=OmegaConf.to_container(cfg.recovery_layer_lrs,resolve=True)
        def setup_groups(module):
            module._optimizer_param_groups=build_groups(module.named_parameters(),rates)
        model.setup_optimizer_param_groups=types.MethodType(setup_groups,model)
    model.setup_optimization(cfg.model.optim)
    if cfg.model.get('spec_augment') is not None:
        model.spec_augmentation=ASRModel.from_config_dict(cfg.model.spec_augment)
    provenance={'source_sha256':SOURCE_SHA,'initial_student_sha256':start_sha,'fits_sha256':sha(cfg.gabor_fits),
        'trainable_parameters':sum(p.numel() for p in model.parameters()),'frozen_rows':receipts,
        'argv':sys.argv,'config':OmegaConf.to_container(cfg,resolve=True),'torch':torch.__version__,
        'temporary_directory':os.environ.get('TMPDIR'),
        'training_sources_sha256':{name:sha(Path(__file__).parent/name) for name in
            ['train.py','fit.py','frozen.py','anchor_loss.py','posterior.py','posterior_loss.py','layer_anchor.py','continuation.py','optimizer_groups.py']}}
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('TRAINABLE_PARAMETERS',provenance['trainable_parameters'],flush=True)
    trainer.fit(model,ckpt_path=cfg.get("resume_checkpoint"))
    for logger in trainer.loggers:logger.finalize('success')
    print('RECOVERY_COMPLETE',flush=True)
    os._exit(0)

if __name__=='__main__':main()
