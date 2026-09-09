#!/bin/bash
# Launch Parakeet-TDT-0.6b-v3 fine-tuning inside the NeMo container.
# Usage: ./train.sh pilot|main [extra hydra overrides...]
set -euo pipefail
MODE="${1:-pilot}"; shift || true
ROOT=/home/nathanroll/parakeet-ft
FIN=$ROOT/manifests/final
BASE=$ROOT/models/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo

case "$MODE" in
  pilot)
    NAME=pilot_el_en
    INPUT_CFG=$FIN/pilot_input_cfg.yaml
    MAX_STEPS=${MAX_STEPS:-600}
    VAL_EVERY=${VAL_EVERY:-200}
    LR=${LR:-2e-5}
    WARMUP=${WARMUP:-100}
    BATCH_DUR=${BATCH_DUR:-200}
    ACCUM=${ACCUM:-2}
    VAL_MANIFESTS="[$FIN/fleurs_el_dev.json,$FIN/libri_en_test_other.json]"
    EXTRA=""
    ;;
  main)
    NAME=${NAME:-main_25lang}
    INPUT_CFG=${INPUT_CFG:-$FIN/train_input_cfg.yaml}
    MAX_STEPS=${MAX_STEPS:-20000}
    VAL_EVERY=${VAL_EVERY:-2000}
    EXTRA="++validate_first=true"
    LR=${LR:-2e-5}
    WARMUP=${WARMUP:-500}
    BATCH_DUR=${BATCH_DUR:-200}
    ACCUM=${ACCUM:-2}
    VAL_MANIFESTS="[$FIN/val_fleurs_dev_sub.json,$FIN/val_cv_dev_sub.json,$FIN/libri_en_test_other.json]"
    ;;
  stage2)
    NAME=${NAME:-stage2_cooldown}
    BASE=${INIT:-$ROOT/models/ft/parakeet-tdt-0.6b-v3-ft-stage1-ema.nemo}
    INPUT_CFG=${INPUT_CFG:-$FIN/stage2_input_cfg.yaml}
    MAX_STEPS=${MAX_STEPS:-4000}
    VAL_EVERY=${VAL_EVERY:-2000}
    LR=${LR:-5e-6}
    WARMUP=${WARMUP:-200}
    BATCH_DUR=${BATCH_DUR:-200}
    ACCUM=${ACCUM:-2}
    VAL_MANIFESTS="[$FIN/val_fleurs_dev_sub.json,$FIN/val_cv_dev_sub.json,$FIN/libri_en_test_other.json]"
    EXTRA="++validate_first=true model.optim.sched.min_lr=5e-7"
    ;;
  *) echo "unknown mode $MODE"; exit 1;;
esac

mkdir -p $ROOT/runs $ROOT/logs
cd $ROOT
NEMO_NAME=train-$NAME exec ./nemo.sh python finetune.py \
  --config-path=$ROOT/Speech/examples/asr/conf/asr_finetune --config-name=speech_to_text_finetune \
  name=$NAME \
  init_from_nemo_model=$BASE \
  model.tokenizer.update_tokenizer=false \
  model.train_ds.manifest_filepath=null \
  ++model.train_ds.use_lhotse=true \
  ++model.train_ds.input_cfg=$INPUT_CFG \
  ++model.train_ds.batch_duration=$BATCH_DUR \
  ++model.train_ds.quadratic_duration=20 \
  ++model.train_ds.use_bucketing=true \
  ++model.train_ds.num_buckets=30 \
  ++model.train_ds.bucket_buffer_size=20000 \
  ++model.train_ds.shuffle_buffer_size=10000 \
  ++model.train_ds.seed=${SEED:-0} \
  ++model.train_ds.shard_seed=randomized \
  ++model.train_ds.pretokenize=false \
  model.train_ds.batch_size=null \
  model.train_ds.max_duration=20 \
  model.train_ds.min_duration=0.4 \
  model.train_ds.num_workers=${NUM_WORKERS:-6} \
  model.train_ds.shuffle=true \
  model.validation_ds.manifest_filepath="$VAL_MANIFESTS" \
  ++model.validation_ds.use_lhotse=true \
  ++model.validation_ds.use_bucketing=false \
  model.validation_ds.batch_size=16 \
  model.validation_ds.num_workers=4 \
  model.test_ds.manifest_filepath=null \
  model.optim.name=adamw \
  model.optim.lr=$LR \
  model.optim.betas='[0.9,0.98]' \
  model.optim.weight_decay=1e-3 \
  model.optim.sched.name=CosineAnnealing \
  model.optim.sched.warmup_steps=$WARMUP \
  ++model.optim.sched.max_steps=$MAX_STEPS \
  model.optim.sched.min_lr=1e-6 \
  model.spec_augment.freq_masks=2 model.spec_augment.time_masks=10 \
  trainer.devices=1 trainer.num_nodes=1 \
  trainer.strategy=auto \
  trainer.precision=bf16-mixed \
  trainer.max_epochs=-1 \
  trainer.max_steps=$MAX_STEPS \
  ++trainer.limit_train_batches=$VAL_EVERY \
  trainer.val_check_interval=$VAL_EVERY \
  trainer.check_val_every_n_epoch=1 \
  ++trainer.use_distributed_sampler=false \
  trainer.accumulate_grad_batches=${ACCUM:-1} \
  trainer.gradient_clip_val=1.0 \
  trainer.log_every_n_steps=20 \
  trainer.num_sanity_val_steps=0 \
  trainer.enable_progress_bar=false \
  exp_manager.exp_dir=$ROOT/runs \
  exp_manager.name=$NAME \
  exp_manager.resume_if_exists=true \
  exp_manager.resume_ignore_no_checkpoint=true \
  ++exp_manager.ema.enable=${EMA:-true} \
  ++exp_manager.ema.decay=0.999 \
  exp_manager.checkpoint_callback_params.monitor=val_wer \
  exp_manager.checkpoint_callback_params.mode=min \
  exp_manager.checkpoint_callback_params.save_top_k=3 \
  exp_manager.checkpoint_callback_params.always_save_nemo=true \
  exp_manager.create_tensorboard_logger=true \
  $EXTRA "$@"
