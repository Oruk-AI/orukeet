import torch
from posterior_loss import sample_indices,gather_steps,tdt_kl


def test_sampling_never_uses_padding():
    torch.manual_seed(10)
    lengths=torch.tensor([1,3,8])
    indices=sample_indices(lengths,40)
    assert (indices>=0).all() and (indices<lengths[:,None]).all()
    features=torch.arange(3*4*8).reshape(3,4,8)
    actual=gather_steps(features,indices)
    for b in range(3):assert torch.equal(actual[b],features[b,:,indices[b]].T)


def test_tdt_distributions_normalize_separately_and_have_gradients():
    torch.manual_seed(11)
    teacher=torch.randn(2,3,4,13)
    student=teacher.clone()
    student[...,:8]+=10
    student[...,8:]-=20
    a,b=tdt_kl(student,teacher,8)
    assert abs(a.item())<1e-6 and abs(b.item())<1e-6
    student=(teacher+torch.randn_like(teacher)*.2).requires_grad_(True)
    a,b=tdt_kl(student,teacher,8)
    assert a>0 and b>0
    (a+b).backward()
    assert torch.isfinite(student.grad).all() and student.grad.abs().sum()>0
