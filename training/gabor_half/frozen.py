"""Exact immutable replacements with ordinary Conv1d exports."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn.utils import parametrize
from fit import reconstruct, SOURCE_SHA

class FrozenGaborRows(nn.Module):
    def __init__(self, weight, rows, parameters):
        super().__init__()
        self.register_buffer('indices',torch.tensor(rows,device=weight.device,dtype=torch.long))
        free=sorted(set(range(weight.shape[0]))-set(rows))
        self.register_buffer('free_indices',torch.tensor(free,device=weight.device,dtype=torch.long))
        self.register_buffer('gabor_parameters',torch.tensor(parameters,dtype=torch.float64,device=weight.device))
        # Synthesize once in float64. Fixed taps avoid trigonometry in every forward.
        values=torch.from_numpy(reconstruct(parameters,weight.shape[-1])).to(weight).unsqueeze(1)
        fixed=torch.zeros_like(weight).index_copy(0,self.indices,values)
        self.register_buffer('fixed',fixed)
    def right_inverse(self, weight):
        return weight.detach().index_select(0,self.free_indices).clone()
    def forward(self, free):
        return self.fixed.index_copy(0,self.free_indices,free)

def install(model, fit_path):
    report=json.loads(Path(fit_path).read_text())
    if report['source_sha256']!=SOURCE_SHA or report['selected']!=12288 or report['kernels']!=24576 or report['debug_limit']:
        raise ValueError('Not a full source-matched 50% fit')
    names={r['name'] for r in report['records']}
    modules=dict(model.named_modules())
    receipts={}
    # Every existing parameter is enabled. The selected rows become buffers.
    model.requires_grad_(True)
    for name in sorted(names):
        module=modules[name.removesuffix('.weight')]
        if not isinstance(module,nn.Conv1d) or module.groups!=1024 or tuple(module.weight.shape)!=(1024,1,9):
            raise ValueError(name)
        records=sorted((r for r in report['records'] if r['name']==name and r['selected']),key=lambda r:r['channel'])
        if not records:continue
        rows=[r['channel'] for r in records];parameters=[r['params'] for r in records]
        p=FrozenGaborRows(module.weight,rows,parameters)
        parametrize.register_parametrization(module,'weight',p,unsafe=True)
        receipts[name]={'count':len(rows),'hash':tensor_sha(module.weight.detach()[p.indices])}
    verify(model,receipts)
    return receipts

def tensor_sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def verify(model, receipts):
    modules=dict(model.named_modules())
    for name,receipt in receipts.items():
        m=modules[name.removesuffix('.weight')];p=m.parametrizations.weight[0]
        if tensor_sha(m.weight[p.indices])!=receipt['hash']:raise RuntimeError('Frozen rows changed: '+name)
        if any(True for _ in p.parameters()):raise RuntimeError('Gabor parameters must be buffers')
    if not all(p.requires_grad for p in model.parameters()):raise RuntimeError('Unexpected frozen non-Gabor parameter')

def materialize(model):
    for module in list(model.modules()):
        if parametrize.is_parametrized(module,'weight') and isinstance(module.parametrizations.weight[0],FrozenGaborRows):
            parametrize.remove_parametrizations(module,'weight',leave_parametrized=True)


def materialized_state_dict(model, state):
    """Replace parametrization internals with portable Conv1d weight keys."""
    for name,module in model.named_modules():
        if parametrize.is_parametrized(module,'weight') and isinstance(module.parametrizations.weight[0],FrozenGaborRows):
            base=name+'.' if name else ''
            prefix=base+'parametrizations.weight.'
            for key in list(state):
                if key.startswith(prefix):del state[key]
            state[base+'weight']=module.weight.detach().clone()
    return state
