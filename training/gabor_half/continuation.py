"""Load an audited dense candidate before installing the same frozen rows."""
import io
import json
from pathlib import Path
import tarfile
import torch
from identity import SOURCE_SHA,sha


def load_candidate(model,path,audit_path):
    path=Path(path);audit=json.loads(Path(audit_path).read_text());digest=sha(path)
    if audit['status']!='pass' or audit['candidate_sha256']!=digest or audit['original_sha256']!=SOURCE_SHA or audit['frozen_gabor_rows_exact']!=12288:
        raise ValueError('Continuation requires an independently audited Gabor candidate')
    with tarfile.open(path) as archive:
        member=next(m for m in archive.getmembers() if m.name.endswith('model_weights.ckpt'))
        state=torch.load(io.BytesIO(archive.extractfile(member).read()),map_location='cpu',weights_only=True)
    model.load_state_dict(state,strict=True)
    return digest
