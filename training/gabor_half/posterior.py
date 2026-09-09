"""Training-only teacher of token and duration probabilities on sampled states."""
import copy
import types
import lightning.pytorch as pl
import torch
from torch import nn
from posterior_loss import sample_indices,gather_steps,tdt_kl


class PosteriorAnchor(pl.Callback):
    def __init__(self,model,scale,asr_scale,points=16):
        if not scale>0 or not asr_scale>0:raise ValueError('Both losses must be active')
        if model.joint.is_adapter_available() or model.joint.masking_prob>0:
            raise ValueError('The teacher recipe requires the unadapted TDT joint')
        self.teacher=nn.ModuleDict({name:copy.deepcopy(module) for name,module in {
            'encoder':model.encoder,'decoder':model.decoder,'enc':model.joint.enc,
            'pred':model.joint.pred,'joint_net':model.joint.joint_net}.items()}).eval().requires_grad_(False)
        self.cache={};self.handles=[];self.parity_checked=False
        if model.joint.log_softmax not in (None,False) or model.joint.temperature!=1.:
            raise ValueError('Expected raw joint logits with temperature 1')
        token_count=model.joint.num_classes_with_blank-model.joint.num_extra_outputs
        if model.joint.num_extra_outputs!=5 or token_count!=8193:raise ValueError('Unexpected source TDT head')

        def hook(name):
            def capture(module,args,kwargs,output):
                if not model.training or not torch.is_grad_enabled():return
                with torch.no_grad():target=self.teacher[name](*args,**kwargs)
                lengths=output[1]+(1 if name=='decoder' else 0)
                indices=sample_indices(lengths,points)
                self.cache[name]=(gather_steps(output[0],indices),gather_steps(target[0],indices))
            return capture
        for name in ('encoder','decoder'):
            self.handles.append(getattr(model,name).register_forward_hook(hook(name),with_kwargs=True))
        original=model.training_step
        def training_step(module,batch,batch_idx):
            self.cache.clear();result=original(batch,batch_idx)
            if set(self.cache)!={'encoder','decoder'}:raise RuntimeError('Missing teacher inputs')
            enc,tenc=self.cache['encoder'];pred,tpred=self.cache['decoder']
            # Match NeMo's standard joint exactly, before its optional softmax.
            logits=model.joint.joint_net(model.joint.enc(enc).unsqueeze(2)+model.joint.pred(pred).unsqueeze(1))
            if not self.parity_checked:
                with torch.no_grad():
                    direct=model.joint.joint_after_projection(model.joint.enc(enc),model.joint.pred(pred))
                if not torch.equal(logits.detach(),direct):raise RuntimeError('Native joint parity failed')
                self.parity_checked=True;print('TEACHER_JOINT_PARITY pass',flush=True)
            with torch.no_grad():
                target=self.teacher['joint_net'](self.teacher['enc'](tenc).unsqueeze(2)+self.teacher['pred'](tpred).unsqueeze(1))
            token_loss,duration_loss=tdt_kl(logits,target,token_count)
            asr=result['loss'] if isinstance(result,dict) else result
            total=asr_scale*asr+scale*(token_loss+duration_loss)
            module.log('teacher_token_kl',token_loss.detach(),on_step=True)
            module.log('teacher_duration_kl',duration_loss.detach(),on_step=True)
            module.log('recovery_total_loss',total.detach(),on_step=True)
            if isinstance(result,dict):result['loss']=total;return result
            return total
        model.training_step=types.MethodType(training_step,model)

    def on_fit_start(self,trainer,model):self.teacher.to(model.device).eval()
