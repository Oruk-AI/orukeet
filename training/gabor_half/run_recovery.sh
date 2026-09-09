#!/usr/bin/env bash
set -euo pipefail
ROOT=${PARAKEET_ROOT:-/home/nathanroll/parakeet-ft}
EXP="$ROOT/gabor_half_20260906"
NAME=${NAME:-orukeet_gabor_half_r1_20260906}
STEPS=${STEPS:-400}
BATCH_SECONDS=${BATCH_SECONDS:-60}
TASK_TMP="$EXP/$NAME-tmp"
mkdir -p "$TASK_TMP"
cd "$ROOT"
# Existing dedicated A100, one GPU. No resources are provisioned by this script.
# fsspec commits through a temporary file. Keep it on the bind-mounted
# filesystem so commit is a rename instead of a second multi-GB copy.
NEMO_NAME="$NAME" ./nemo.sh env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 TMPDIR="$TASK_TMP" python "$EXP/train.py" \
 --config-path="$ROOT/Speech/examples/asr/conf/asr_finetune" --config-name=speech_to_text_finetune \
 name="$NAME" init_from_nemo_model="$ROOT/models/ft/stage3_baseblend_a075_20260905.nemo" \
 model.tokenizer.update_tokenizer=false ++seed=20260906 \
 ++gabor_fits="$EXP/fit-full/fits.json" ++recovery_output="$EXP/$NAME" ++export_interval=100 \
 model.train_ds.manifest_filepath=null ++model.train_ds.use_lhotse=true \
 ++model.train_ds.input_cfg="$ROOT/manifests/balanced_global_20260905/input_cfg_with_accents.yaml" \
 ++model.train_ds.batch_duration="$BATCH_SECONDS" ++model.train_ds.quadratic_duration=20 \
 ++model.train_ds.use_bucketing=true ++model.train_ds.num_buckets=20 \
 ++model.train_ds.bucket_buffer_size=10000 ++model.train_ds.shuffle_buffer_size=10000 \
 ++model.train_ds.seed=20260906 ++model.train_ds.shard_seed=20260906 ++model.train_ds.pretokenize=false \
 model.train_ds.batch_size=null model.train_ds.max_duration=20 model.train_ds.min_duration=0.4 \
 model.train_ds.num_workers=4 model.train_ds.shuffle=true \
 model.validation_ds.manifest_filepath=null model.test_ds.manifest_filepath=null \
 model.optim.name=adamw model.optim.lr=2e-6 model.optim.betas='[0.9,0.98]' model.optim.weight_decay=1e-3 \
 model.optim.sched.name=CosineAnnealing model.optim.sched.warmup_steps=25 \
 ++model.optim.sched.max_steps="$STEPS" model.optim.sched.min_lr=5e-7 \
 model.spec_augment.freq_masks=2 model.spec_augment.time_masks=2 \
 trainer.devices=1 trainer.num_nodes=1 trainer.strategy=auto trainer.precision=bf16-mixed \
 trainer.max_epochs=-1 trainer.max_steps="$STEPS" ++trainer.limit_train_batches=$((STEPS * 4)) \
 ++trainer.limit_val_batches=0 trainer.check_val_every_n_epoch=1 ++trainer.use_distributed_sampler=false \
 trainer.accumulate_grad_batches=4 trainer.gradient_clip_val=1.0 trainer.log_every_n_steps=10 \
 trainer.num_sanity_val_steps=0 trainer.enable_progress_bar=false ++trainer.enable_checkpointing=false \
 exp_manager.exp_dir="$ROOT/runs" exp_manager.name="$NAME" exp_manager.resume_if_exists=false \
 exp_manager.resume_ignore_no_checkpoint=true ++exp_manager.ema.enable=false \
 exp_manager.create_checkpoint_callback=false exp_manager.create_tensorboard_logger=true "$@"
