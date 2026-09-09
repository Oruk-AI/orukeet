#!/bin/bash
# Wrapper: run a command inside the NeMo Speech container with GPU + project mounts.
# Runs as root (the image's python lives under /root); outputs are chown'ed back afterwards
# for the directories listed in NEMO_CHOWN (space-separated, default: none).
# Usage: ./nemo.sh python script.py args...      (cwd inside container = /home/nathanroll/parakeet-ft)
sudo docker run --rm ${NEMO_TTY:--i} --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --shm-size=16g \
  -v /home/nathanroll:/home/nathanroll \
  -v /work:/work \
  -e HF_HOME=/home/nathanroll/.cache/huggingface \
  -e PYTHONUNBUFFERED=1 -e TOKENIZERS_PARALLELISM=false \
  -e WANDB_MODE=${WANDB_MODE:-disabled} \
  -e NEMO_CACHE_DIR=/home/nathanroll/.cache/torch/NeMo \
  -e PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True} \
  -w /home/nathanroll/parakeet-ft \
  --name ${NEMO_NAME:-nemo-$$} \
  nvcr.io/nvidia/nemo-speech:26.07.00 "$@"
rc=$?
for d in ${NEMO_CHOWN:-}; do sudo chown -R nathanroll:nathanroll "$d" 2>/dev/null; done
exit $rc
