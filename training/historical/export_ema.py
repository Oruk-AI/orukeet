"""Load EMA weights from a Lightning ckpt into the base architecture and save a .nemo.
usage: python export_ema.py CKPT OUT.nemo"""
import sys, torch
from nemo.collections.asr.models import ASRModel

BASE = "/home/nathanroll/parakeet-ft/models/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo"
ckpt, out = sys.argv[1], sys.argv[2]
model = ASRModel.restore_from(BASE, map_location="cpu")
sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
missing, unexpected = model.load_state_dict(sd, strict=True)
model.save_to(out)
print("EXPORTED", out)
