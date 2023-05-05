import sys
from dataclasses import dataclass
import numpy as np
import jax
from xbarax.xla_crossbar_interface_singleBuf.custom_xla_matmul import get_mcbmm_fn
from xbarax.xla_crossbar_interface_singleBuf.custom_xla_add import get_mcbadd_fn

from xbarax.xla_crossbar_interface_singleBuf.custom_xla_array import MemristiveCrossbarArray, AbstractMemristiveCrossbarArray, set_memristive_crossbar_array_device_put_handler

libfile = "../xbarax/xla_crossbar_interface_singleBuf/libfuncs.so"

mcbmm = get_mcbmm_fn(libfile)
mcbadd = get_mcbadd_fn(libfile)

set_memristive_crossbar_array_device_put_handler(libfile)
AbstractMemristiveCrossbarArray.set_matmul_fn(mcbmm)
AbstractMemristiveCrossbarArray.set_add_fn(mcbadd)

def calc_loss(mat, x):
    y = mat @ x
    return y.sum()

def example_fn(mat, x):
    grad = jax.grad(calc_loss)(mat, x)
    update = mat + grad
    return update


dim_in  = 5
dim_out = 9
# a_ref = np.repeat(np.arange(dim_out), dim_in).reshape((dim_out, dim_in)).astype(np.float32)
a_ref = np.random.random((dim_out, dim_in)).astype(np.float32)
a = MemristiveCrossbarArray(a_ref.copy())
b = np.random.random((dim_in, )).astype(np.float32)

from jax import make_jaxpr

target = np.random.random((dim_out,)).astype(np.float32)


print("\n==========================================================")
print("            jaxpr of example_fn without jit:")
print("==========================================================\n")
print(f"------ type(max)=={type(a_ref)} --------")
print(make_jaxpr(example_fn)(a_ref, b).jaxpr)
print()
print(f"------ type(max)=={type(a)} --------")
print(make_jaxpr(example_fn)(a, b).jaxpr)

print("\n==========================================================")
print("              jaxpr of example_fn with jit:")
print("==========================================================\n")
print(f"------ type(max)=={type(a_ref)} --------")
print(make_jaxpr(jax.jit(example_fn))(a_ref, b).jaxpr)
print()
print(f"------ type(max)=={type(a)} --------")
print(make_jaxpr(jax.jit(example_fn))(a, b).jaxpr)
print("==========================================================")

