#!/usr/bin/env python3
"""Conservative continuation: upper encoder adaptation with a frozen teacher.

The previous decoder, vocabulary and BatchNorm statistics are retained. A
cosine-distance penalty anchors encoder representations to the initialization
on exactly the same (possibly augmented) features used by the student.
Teacher tensors are owned by the callback, never saved in the exported model.
"""
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

ROOT=Path(os.environ.get("PARAKEET_ROOT","/home/nathanroll/parakeet-ft"))
sys.path.insert(0,str(ROOT/"Speech/examples/asr"))
from speech_to_text_finetune import get_base_model, check_vocabulary, setup_dataloaders
from nemo.collections.asr.models import ASRModel
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg


class RepresentationAnchor(pl.Callback):
    def __init__(self, scale):
        self.scale=float(scale)
        self.teacher=None
        self.anchor_loss=None

    def install(self, model, top_layers):
        self.teacher=copy.deepcopy(model.encoder).eval()
        self.teacher.requires_grad_(False)
        model.requires_grad_(False)
        for layer in model.encoder.layers[-top_layers:]: layer.requires_grad_(True)
        original_train=model.train
        def train(mode=True):
            original_train(mode)
            if mode:
                model.preprocessor.eval()
                model.encoder.pre_encode.eval()
                model.decoder.eval()
                model.joint.eval()
                for layer in model.encoder.layers[:-top_layers]: layer.eval()
                for layer in model.encoder.modules():
                    if isinstance(layer,_BatchNorm):layer.eval()
            return model
        model.train=train
        model.train()
        def hook(module,args,kwargs,output):
            if not model.training or not torch.is_grad_enabled():return
            with torch.no_grad():
                teacher,_=self.teacher(*args,**kwargs)
            student,lengths=output
            sim=(F.normalize(student.float(),dim=1)*F.normalize(teacher.float(),dim=1)).sum(dim=1)
            mask=torch.arange(sim.shape[1],device=sim.device)[None,:]<lengths[:,None]
            self.anchor_loss=((1-sim)*mask).sum()/mask.sum().clamp_min(1)
        self.handle=model.encoder.register_forward_hook(hook,with_kwargs=True)
        original_step=model.training_step
        def step(module,batch,batch_idx):
            self.anchor_loss=None
            result=original_step(batch,batch_idx)
            if self.anchor_loss is None:raise RuntimeError("Teacher anchor did not run")
            loss=result["loss"] if isinstance(result,dict) else result
            anchored=loss+self.scale*self.anchor_loss
            module.log("train_anchor_cosine",self.anchor_loss.detach(),on_step=True)
            if isinstance(result,dict):result["loss"]=anchored;return result
            return anchored
        model.training_step=types.MethodType(step,model)
        print("TRAINABLE_PARAMETERS",sum(p.numel() for p in model.parameters() if p.requires_grad),flush=True)

    def on_fit_start(self,trainer,pl_module):
        self.teacher.to(pl_module.device).eval()

    def on_train_batch_start(self,trainer,pl_module,batch,batch_idx):
        for module in pl_module.encoder.modules():
            if isinstance(module,_BatchNorm):module.eval()


@hydra_runner(config_path="conf/asr_finetune",config_name="speech_to_text_finetune")
def main(cfg):
    pl.seed_everything(int(cfg.get("seed",20260905)),workers=True)
    anchor=RepresentationAnchor(cfg.get("anchor_scale",0.1))
    trainer=pl.Trainer(**resolve_trainer_cfg(cfg.trainer),callbacks=[anchor])
    exp_manager(trainer,cfg.get("exp_manager"))
    model=check_vocabulary(get_base_model(trainer,cfg),cfg)
    n=int(cfg.get("train_top_layers",6))
    if not 1<=n<=len(model.encoder.layers):raise ValueError("Invalid top-layer count")
    anchor.install(model,n)
    model=setup_dataloaders(model,cfg)
    model.setup_optimization(cfg.model.optim)
    if cfg.model.get("spec_augment") is not None:
        model.spec_augmentation=ASRModel.from_config_dict(cfg.model.spec_augment)
    trainer.fit(model)
    dest=Path(cfg.get("export_path",str(ROOT/"models/ft/anchored_20260905.nemo")))
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():raise FileExistsError(dest)
    model.save_to(str(dest))
    for logger in trainer.loggers:
        logger.finalize("success")
    print("TRAINING_COMPLETE",dest,flush=True)
    # This pinned NeMo container repeatedly hangs while joining data workers
    # after successful export. All checkpoints and loggers are closed above;
    # exit the dedicated job process so its GPU allocation is released.
    if cfg.get("exit_after_export",True):os._exit(0)


if __name__=="__main__":main()
