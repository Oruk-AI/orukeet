#!/usr/bin/env python3
"""Quantize a float FastConformer encoder ONNX graph for ONNX Runtime on Apple silicon (M2..M5).

Replaces onnxruntime.quantization.quantize_dynamic(weight_type=QUInt8) for the encoder. Every 2-D MatMul with a
constant weight, and every 1x1 (pointwise) Conv, becomes a com.microsoft.DynamicQuantizeMatMul node with
symmetric int8 per-output-channel weights (zero point 0) and the bias folded in. That is the form ONNX Runtime
routes to KleidiAI's SME2 int8 kernels on M4/M5 (and to the I8MM/UDOT NEON kernels elsewhere); asymmetric uint8
weights are never eligible. Each layer's Q/K/V projections are merged into one N=3072 GEMM followed by a Split
(small-N GEMMs are inefficient on SME). Depthwise 9-tap convs and the 2-D subsampling convs stay in float.

Usage: quantize_encoder_sme.py encoder.onnx encoder.int8.onnx [--no-merge-qkv] [--no-pointwise]
"""
import argparse, onnx, numpy as np, collections
from onnx import numpy_helper, helper, TensorProto
ap=argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("dst")
ap.add_argument("--no-merge-qkv",action="store_true"); ap.add_argument("--no-pointwise",action="store_true"); a=ap.parse_args()
m=onnx.load(a.src, load_external_data=True); g=m.graph; init={t.name:t for t in g.initializer}
cons=collections.defaultdict(list)
for n in g.node:
    for i in n.input: cons[i].append(n)
def qsym_pc(Wf):
    amax=np.abs(Wf).max(axis=0); s=np.where(amax>0,amax/127.0,1.0).astype(np.float32)
    return np.clip(np.rint(Wf/s),-127,127).astype(np.int8), s
def add_init(name,arr): g.initializer.append(numpy_helper.from_array(arr,name)); init[name]=g.initializer[-1]
def bias_after(out_name, N):
    """If out_name feeds exactly one Add with a constant [N] (or [1,1,N]) bias, return (add_node, bias_array)."""
    c=cons.get(out_name,[])
    if len(c)!=1 or c[0].op_type!="Add": return None
    add=c[0]; other=[i for i in add.input if i!=out_name]
    if len(other)!=1 or other[0] not in init: return None
    b=numpy_helper.to_array(init[other[0]]).astype(np.float32)
    if b.size!=N: return None
    return add, b.reshape(N)
remove=set(); new=[]; stats=collections.Counter()
for n in g.node:
    if n.op_type=="MatMul" and n.input[1] in init and len(init[n.input[1]].dims)==2 and init[n.input[1]].data_type==TensorProto.FLOAT:
        Wf=numpy_helper.to_array(init[n.input[1]]); K,N=Wf.shape
        Wq,s=qsym_pc(Wf); base=n.input[1]
        add_init(base+"_q8",Wq); add_init(base+"_scale",s); add_init(base+"_zp",np.array(0,dtype=np.int8))
        out=n.output[0]; inputs=[n.input[0],base+"_q8",base+"_scale",base+"_zp"]; ba=bias_after(out,N)
        if ba: add,b=ba; add_init(base+"_bias",b); inputs.append(base+"_bias"); out=add.output[0]; remove.add(id(add)); stats["bias folded"]+=1
        new.append(helper.make_node("DynamicQuantizeMatMul",inputs,[out],name=n.name+"_dqmm",domain="com.microsoft")); stats["MatMul -> DynamicQuantizeMatMul"]+=1; continue
    if not a.no_pointwise and n.op_type=="Conv" and n.input[1] in init and len(init[n.input[1]].dims)==3 and init[n.input[1]].dims[2]==1 \
       and all(helper.get_attribute_value(at)==1 for at in n.attribute if at.name=="group"):
        W=numpy_helper.to_array(init[n.input[1]]); Cout,Cin=W.shape[0],W.shape[1]
        Wq,s=qsym_pc(W.reshape(Cout,Cin).T.copy()); base=n.input[1]
        add_init(base+"_q8",Wq); add_init(base+"_scale",s); add_init(base+"_zp",np.array(0,dtype=np.int8))
        inputs=[n.output[0]+"_xT",base+"_q8",base+"_scale",base+"_zp"]
        if len(n.input)>2 and n.input[2] in init: inputs.append(n.input[2]); stats["conv bias folded"]+=1
        new.append(helper.make_node("Transpose",[n.input[0]],[n.output[0]+"_xT"],perm=[0,2,1],name=n.name+"_xT"))
        new.append(helper.make_node("DynamicQuantizeMatMul",inputs,[n.output[0]+"_yT"],name=n.name+"_dqmm",domain="com.microsoft"))
        new.append(helper.make_node("Transpose",[n.output[0]+"_yT"],[n.output[0]],perm=[0,2,1],name=n.name+"_yT")); stats["pointwise Conv -> DynamicQuantizeMatMul"]+=1; continue
    new.append(n)
new=[n for n in new if id(n) not in remove]
if not a.no_merge_qkv:
    byA=collections.defaultdict(list)
    for n in new:
        if n.op_type=="DynamicQuantizeMatMul" and any(k in n.name for k in ("linear_q","linear_k","linear_v")): byA[n.input[0]].append(n)
    merged=[]; drop=set()
    for A,nodes in byA.items():
        if len(nodes)!=3: continue
        nodes.sort(key=lambda n: ["linear_q","linear_k","linear_v"].index(next(k for k in ("linear_q","linear_k","linear_v") if k in n.name)))
        W=np.concatenate([numpy_helper.to_array(init[n.input[1]]) for n in nodes],axis=1); S=np.concatenate([numpy_helper.to_array(init[n.input[2]]) for n in nodes])
        Bs=[numpy_helper.to_array(init[n.input[4]]) if len(n.input)>4 else np.zeros(W.shape[1]//3,np.float32) for n in nodes]; B=np.concatenate(Bs)
        base=nodes[0].name.replace("linear_q","linear_qkv"); add_init(base+"_q8",W); add_init(base+"_scale",S.astype(np.float32)); add_init(base+"_zp",np.array(0,dtype=np.int8)); add_init(base+"_bias",B.astype(np.float32))
        merged.append((A,[helper.make_node("DynamicQuantizeMatMul",[A,base+"_q8",base+"_scale",base+"_zp",base+"_bias"],[base+"_Y"],name=base,domain="com.microsoft"),
                          helper.make_node("Split",[base+"_Y"],[n.output[0] for n in nodes],axis=-1,name=base+"_split")]))
        drop.update(id(n) for n in nodes); stats["QKV merged"]+=1
    new=[n for n in new if id(n) not in drop]
    for A,(mm,sp) in merged:
        pos=max([i for i,n in enumerate(new) if A in n.output]+[-1])+1; new.insert(pos,sp); new.insert(pos,mm)
del g.node[:]; g.node.extend(new)
keep=set(i for n in new for i in n.input)|set(o.name for o in g.output)
for t in list(g.initializer):
    if t.name not in keep: g.initializer.remove(t)
if not any(o.domain=="com.microsoft" for o in m.opset_import): m.opset_import.append(helper.make_opsetid("com.microsoft",1))
for k,v in stats.items(): print(f"  {k}: {v}")
print(f"nodes: {len(g.node)}"); onnx.save(m,a.dst)
