import sys
from dataclasses import dataclass
import numpy as np
import jax
jax.config.update('jax_platform_name', 'cpu') # use cpu (xbar backend is only defined for cpu host)
from xbarax.xla_crossbar_interface_singleBuf.custom_xla_matmul import get_mcbmm_fn
from xbarax.xla_crossbar_interface_singleBuf.custom_xla_add import get_mcbadd_fn

from xbarax.xla_crossbar_interface_singleBuf.custom_xla_array import MemristiveCrossbarArray, AbstractMemristiveCrossbarArray, set_memristive_crossbar_array_device_put_handler

import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath("")), "..", "xbarax"))

libfile = "../xbarax/xla_crossbar_interface_singleBuf/libfuncs.so"

mcbmm = get_mcbmm_fn(libfile)
mcbadd = get_mcbadd_fn(libfile)

set_memristive_crossbar_array_device_put_handler(libfile)
AbstractMemristiveCrossbarArray.set_matmul_fn(mcbmm)
AbstractMemristiveCrossbarArray.set_add_fn(mcbadd)

def matmul(a, b):
    return a @ b

def add(a, b): 
    return a + b

if __name__ == '__main__':
    # a = np.ones((4, 5), dtype=np.float32)
    dim_in  = 5
    dim_out = 9
    # a_ref = np.repeat(np.arange(dim_out), dim_in).reshape((dim_out, dim_in)).astype(np.float32)
    a_ref = np.random.random((dim_out, dim_in)).astype(np.float32)
    a = MemristiveCrossbarArray(a_ref.copy())

    a_2 = MemristiveCrossbarArray(a_ref.copy())
    b = np.random.random((dim_in, )).astype(np.float32)

    print("\n---------------------------------------------------------------")
    print("\na_ref")
    print(type(a_ref))
    print(a_ref)
    print("\na")
    print(type(a))
    print(a)

    # usign a will result in it being allocated on the device
    print("\n---------------------------------------------------------------")
    print("\nSimple matmul")
    y = a @ b
    print("a", a.is_set, a.is_deallocated, a.use_crossbar)


    print("\n---------------------------------------------------------------")
    print("Using the memristive crossbar in an addtion will deallocate the original array and set the result")
    try:
        print("\nbefore add:")
        print("a is_set", a.is_set)
        # use the crossbar in an add operation
        a_new = a + np.zeros(a.shape)
        print("after add:")
        print("a is_set", a.is_set)
        print("a_new is_set", a_new.is_set)

        # trying to use a now will result in an error
        y = a @ b
    except Exception as e:
        print("\nException")
        print(e)
        a = a_new


    print("\n---------------------------------------------------------------")
    print("some typical usecases:")
    print("standard execution: ", matmul(a, b))
    print("jit:                ", jax.jit(matmul)(a, b))
    print("jit:                ", jax.jit(matmul)(a, b))
    jvp_res = jax.jvp(mcbmm, (a, b), (np.ones_like(a.conductences), np.ones_like(b)))
    print("jvp: primals:        ", jvp_res[0])
    print("jvp: jvp:           ", jvp_res[1])
    jvp_res_jit = jax.jit(lambda arg_values, arg_tangents: 
                   jax.jvp(mcbmm, arg_values, arg_tangents))(
                    (a, b), (np.ones_like(a.conductences), np.ones_like(b)))
    print("jit jvp: output:    ", jvp_res_jit[0])
    print("jit jvp: jvp:       ", jvp_res_jit[1])



    print("\n---------------------------------------------------------------")
    print("`MemristiveCrossbarArry` is compatible with all of jax function transformations like `jax.jit`, `jax.grad` and `jax.vmap`")
    num_batches = 3
    b_batched = np.arange(b.size*num_batches, dtype=np.float32).reshape((*b.shape, num_batches))

    sum_val, grad_of_sum = jax.jit(jax.vmap(jax.value_and_grad(lambda x, y: (x @ y).sum(), argnums=(0, 1)), in_axes=(None, 1), out_axes=0))(a, b_batched)
    sum_val_ref, grad_of_sum_ref = jax.jit(jax.vmap(jax.value_and_grad(lambda x, y: (x @ y).sum(), argnums=(0, 1)), in_axes=(None, 1), out_axes=0))(a_ref, b_batched)
    print("\nmemristive corssbar array")
    print("jit vmap grad:(prim)", sum_val)
    print("jit vmap grad:(grad)", grad_of_sum)
    print("\njax array reference")
    print("jit vmap grad:(prim)", sum_val_ref)
    print("jit vmap grad:(grad)", grad_of_sum_ref)
    print("\ncomparison")
    if np.allclose(sum_val,sum_val_ref):
        print(f"\u001b[32m jit vmap grad:(prim): Success!\u001b[0m")
    else:
        print(f"\u001b[31m jit vmap grad:(prim): Wrong results!\u001b[0m")
    if all(jax.tree_util.tree_leaves(jax.tree_util.tree_map(np.allclose, grad_of_sum, grad_of_sum_ref))):
        print(f"\u001b[32m jit vmap grad:(grad): Success!\u001b[0m")
    else:
        print(f"\u001b[31m jit vmap grad:(grad): Wrong results!\u001b[0m")

    print("\n---------------------------------------------------------------")
    print("Is is also compatible with `jax.tree_util.tree_map`")
    print("a", a)
    print("a_ref", a_ref)
    res = jax.tree_map(lambda x, y: x+y, a, a_ref)
    print(res)