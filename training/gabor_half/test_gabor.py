import numpy as np
import torch
from torch import nn
from torch.nn.utils import parametrize
from fit import basis, projected, fit_one, reconstruct, generate_bank, choose_starts
from frozen import FrozenGaborRows, materialize, materialized_state_dict

def test_projection_jacobian():
    rng=np.random.default_rng(17);y=rng.normal(size=9);p=np.array([.4,np.log(1.3),.17])
    analytic=projected(p,y,True)
    for j in range(3):
        step=np.zeros(3);step[j]=1e-6
        np.testing.assert_allclose(analytic[:,j],(projected(p+step,y)-projected(p-step,y))/2e-6,rtol=2e-5,atol=2e-6)

def test_signed_fit_and_channel_mapping():
    params=np.array([[1.3,-1.1,2.3,.2,2.8],[.6,1.3,1.4,.39,-2.1]])
    targets=reconstruct(params);grid,u=generate_bank()
    for y in targets:
        fitted,error,_=fit_one((y,choose_starts(y,grid,u)))
        assert error<1e-12
        np.testing.assert_allclose(reconstruct([fitted])[0],y,atol=1e-6)
    conv=nn.Conv1d(4,4,9,padding=4,groups=4,bias=True)
    original=conv.weight.detach().clone()
    frozen=FrozenGaborRows(conv.weight,[0,3],params)
    parametrize.register_parametrization(conv,'weight',frozen,unsafe=True)
    assert conv.parametrizations.weight.original.shape==(2,1,9)
    assert list(frozen.parameters())==[]
    torch.testing.assert_close(conv.weight[[1,2]],original[[1,2]],rtol=0,atol=0)
    expected=conv.weight[[0,3]].detach().clone()
    optim=torch.optim.AdamW(conv.parameters(),lr=.02,weight_decay=.7)
    for _ in range(5):
        optim.zero_grad();conv(torch.randn(3,4,20)).square().mean().backward();optim.step()
        torch.testing.assert_close(conv.weight[[0,3]],expected,rtol=0,atol=0)
    assert not torch.equal(conv.weight[[1,2]],original[[1,2]])
    x=torch.randn(2,4,25);before=conv(x).detach()
    saved=conv.state_dict()
    clone=nn.Conv1d(4,4,9,padding=4,groups=4,bias=True)
    parametrize.register_parametrization(clone,'weight',FrozenGaborRows(clone.weight,[0,3],params),unsafe=True)
    clone.load_state_dict(saved)
    torch.testing.assert_close(clone(x),before,rtol=0,atol=0)
    portable=materialized_state_dict(conv,conv.state_dict())
    vanilla=nn.Conv1d(4,4,9,padding=4,groups=4,bias=True)
    vanilla.load_state_dict(portable,strict=True)
    torch.testing.assert_close(vanilla(x),before,rtol=0,atol=0)
    materialize(conv)
    assert set(conv.state_dict())=={'weight','bias'}
    torch.testing.assert_close(conv(x),before,rtol=0,atol=0)
    torch.testing.assert_close(conv.weight[[0,3]],expected,rtol=0,atol=0)
