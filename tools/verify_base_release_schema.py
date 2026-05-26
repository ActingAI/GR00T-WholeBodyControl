#!/usr/bin/env python3
from pathlib import Path
import sys, onnx
def dims(v): return [d.dim_value or d.dim_param for d in v.type.tensor_type.shape.dim]
def check(p,inp,out,name):
 m=onnx.load(p); onnx.checker.check_model(m); ins=[(i.name,dims(i)) for i in m.graph.input]; outs=[(o.name,dims(o)) for o in m.graph.output]; print(p, ins, outs); assert ins==[("obs_dict",inp)], ins; assert outs==[(name,out)], outs
b=Path(sys.argv[1])
check(b/"model_encoder.onnx", [1,1762], [1,64], "encoded_tokens")
check(b/"model_decoder.onnx", [1,994], [1,29], "action")
print("OK base public .pt release-schema bundle")
