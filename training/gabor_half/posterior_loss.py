"""Sample valid transducer lattice states and preserve both TDT distributions."""
import torch
import torch.nn.functional as F


def sample_indices(lengths, count):
    if bool((lengths <= 0).any()):raise ValueError('Empty sequence')
    return (torch.rand(len(lengths),count,device=lengths.device)*lengths[:,None]).long()


def gather_steps(features, indices):
    # Features arrive in NeMo's B,D,T convention.
    return features.transpose(1,2).gather(1,indices[:,:,None].expand(-1,-1,features.shape[1]))


def tdt_kl(student, teacher, token_count):
    if student.shape!=teacher.shape or not 0<token_count<student.shape[-1]:
        raise ValueError('TDT token/duration partition mismatch')
    losses=[]
    for sl in (slice(None,token_count),slice(token_count,None)):
        logp=F.log_softmax(student[...,sl].float(),dim=-1)
        logq=F.log_softmax(teacher[...,sl].float(),dim=-1)
        losses.append((logq.exp()*(logq-logp)).sum(dim=-1).mean())
    return losses
