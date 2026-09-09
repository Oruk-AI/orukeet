"""Parakeet-TDT-0.6b-v3 fine-tuning entrypoint.

Thin wrapper around NeMo's examples/asr/speech_to_text_finetune.py with two fixes:

1. Frozen BatchNorm statistics in the Conformer encoder. The v3 checkpoint's BN running
   stats do not match its activation statistics; in train() mode BN switches to batch
   stats and the model collapses (loss 2.8 -> 11.7, val WER 100%). We keep BN layers in
   eval mode during training (affine params still train) - see diag4.py.
2. SpecAugment override actually applied (stock script assigns `spec_augment` but the
   model reads `spec_augmentation`).
"""
import sys
import torch
import lightning.pytorch as pl
from omegaconf import OmegaConf
from torch.nn.modules.batchnorm import _BatchNorm

sys.path.insert(0, "/home/nathanroll/parakeet-ft/Speech/examples/asr")
from speech_to_text_finetune import get_base_model, check_vocabulary, setup_dataloaders  # noqa: E402

from nemo.collections.asr.models import ASRModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg


def freeze_bn_stats(model):
    bns = [m for m in model.encoder.modules() if isinstance(m, _BatchNorm)]
    orig_train = model.train

    def train(mode: bool = True):
        orig_train(mode)
        if mode:
            for m in bns:
                m.eval()
        return model

    model.train = train
    model.train()
    logging.info(f"[finetune] frozen running stats of {len(bns)} encoder BatchNorm layers")


class BNGuard(pl.Callback):
    """Belt and braces: re-assert BN eval mode at every train batch start."""

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        for m in pl_module.encoder.modules():
            if isinstance(m, _BatchNorm) and m.training:
                m.eval()


@hydra_runner(config_path="conf/asr_finetune", config_name="speech_to_text_finetune")
def main(cfg):
    logging.info(f"Hydra config: {OmegaConf.to_yaml(cfg)}")
    trainer = pl.Trainer(**resolve_trainer_cfg(cfg.trainer), callbacks=[BNGuard()])
    exp_manager(trainer, cfg.get("exp_manager", None))

    asr_model = get_base_model(trainer, cfg)
    asr_model = check_vocabulary(asr_model, cfg)
    asr_model = setup_dataloaders(asr_model, cfg)
    asr_model.setup_optimization(cfg.model.optim)

    if cfg.model.get("spec_augment") is not None:
        asr_model.spec_augmentation = ASRModel.from_config_dict(cfg.model.spec_augment)
        logging.info(f"[finetune] spec_augment: {OmegaConf.to_container(cfg.model.spec_augment)}")

    if cfg.get("freeze_bn", True):
        freeze_bn_stats(asr_model)

    if cfg.get("validate_first", False) and not trainer.ckpt_path:
        # step-0 metrics of the base model so in-training val_wer trends are interpretable
        trainer.validate(asr_model)
        logging.info(f"[finetune] step-0 validation: {trainer.callback_metrics}")

    trainer.fit(asr_model)


if __name__ == "__main__":
    main()  # noqa pylint: disable=no-value-for-parameter
