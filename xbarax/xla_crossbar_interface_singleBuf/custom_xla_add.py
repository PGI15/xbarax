import sys
import ctypes

import functools as ft
import numpy as np
import jax
import jax.numpy as jnp

from jax import core
from jax._src import abstract_arrays
from jax.interpreters import xla
from jax._src.lib import xla_client
from jax.interpreters import ad
from jax.interpreters import batching

# from jax.lib import xla_client
# xla_client.register_custom_call_target(b"cpu_add", cpu_add_fn)

def get_mcbadd_fn(so_file):

    def mcbadd(a, b):
        """The JAX-traceable way to use the JAX primitive.
        
        Note that the traced arguments must be passed as positional arguments
        to `bind`. 
        """
        return _mcbadd_p.bind(a, b)

    def _mcbadd_abstract_eval(a, b):
        assert len(a.shape) == 2
        assert b.shape == a.shape
        assert a.dtype == np.dtype("float32")
        assert b.dtype == np.dtype("float32")
        ret = type(a)(a.shape, a.dtype, was_set=a.was_set)
        # print("ret")
        # print(ret)
        # print(ret._is_deallocated)
        # ret._is_deallocated = False
        # print(ret._is_deallocated)
        return ret 

    def _mcbadd_xla_translation(ctx, avals_in, avals_out, ac, bc):
        # The inputs have "shapes" that provide both the shape and the dtype

        ac_shape = avals_in[0]
        bc_shape = avals_in[1]
        cc_shape = avals_out[0]
        # print(avals_out)
        # sys.exit()

        # Extract the dtype and shape
        dtype = ac_shape.dtype
        dims = ac_shape.shape
        shape_tuple = ac_shape.shape
        assert len(dims) == 2
        assert bc_shape.dtype == dtype
        assert bc_shape.shape == shape_tuple
        assert cc_shape.dtype == dtype
        assert cc_shape.shape == shape_tuple

        dims = xla_client.ops.ConstantLiteral(ctx.builder, np.asarray(shape_tuple, dtype=np.uint64))

        shape_dims = xla_client.Shape.array_shape(np.dtype(np.uint64), (2,), (0,))

        shape = xla_client.Shape.array_shape(
                np.dtype(ac_shape.dtype), ac_shape.shape, tuple(range(len(ac_shape.shape) - 1, -1, -1))
        )

        # print("ac")
        # print(ac)
        # print("ac_shape")
        # print(ac_shape)
        # print(ac_shape.is_deallocated)
        # print("out shape")
        # print(shape)
        # print("cc_shape")
        # print(cc_shape)
        # print("pre", cc_shape.is_deallocated)
        ac_shape._deallocate()
        # print("post", cc_shape.is_deallocated)
        # print(id(ac_shape))
        # print(id(cc_shape))


        # sys.exit()

        # We dispatch a different call depending on the dtype
        if dtype == np.float32:
            op_name = b"cpu_mcbadd_f32"
        # elif dtype == np.float64:
        #     op_name = platform.encode() + b"_kepler_f64"
        else:
            raise NotImplementedError(f"Unsupported dtype {dtype}")

        # We pass the size of the data as a the first input argument
        return [xla_client.ops.CustomCallWithLayout(
            builder=ctx.builder,
            call_target_name=op_name,
            operands=(dims, ac, bc),
            shape_with_layout=shape, #array_shapes[2][0],
            # shape_with_layout=xla_client.Shape.tuple_shape(array_shapes[2]), #array_shapes[2][0],
            operand_shapes_with_layout=(
                shape_dims,
                shape,
                shape,
            ),
        )]
        
    def _mcbadd_value_and_jvp(arg_values, arg_tangents):
        """Evaluates the primal output and the tangents (Jacobian-vector product).

        Given values of the arguments and perturbation of the arguments (tangents), 
        compute the output of the primitive and the perturbation of the output.

        This method must be JAX-traceable. JAX may invoke it with abstract values 
        for the arguments and tangents.

        Args:
            arg_values: a tuple of arguments
            arg_tangents: a tuple with the tangents of the arguments. The tuple has 
            the same length as the arg_values. Some of the tangents may also be the 
            special value ad.Zero to specify a zero tangent.
        Returns:
            a pair of the primal output and the tangent.
        """
        a, b = arg_values
        at, bt = arg_tangents

        # Now we have a JAX-traceable computation of the output. 
        # Normally, we can use the ma primtive itself to compute the primal output. 
        primal_out = mcbadd(a, b)
        
        #   # We must use a JAX-traceable way to compute the tangent. It turns out that 
        #   # the output tangent can be computed as (xt * y + x * yt + zt),
        #   # which we can implement in a JAX-traceable way using the same "multiply_add_prim" primitive.
        
        # We do need to deal specially with Zero. Here we just turn it into a 
        # proper tensor of 0s (of the same shape as 'x'). 
        # An alternative would be to check for Zero and perform algebraic 
        # simplification of the output tangent computation.
        # def make_zero(tan, likearr):
        #     return jax.lax.zeros_like_array(likearr) if type(tan) is ad.Zero else tan  

        # at_z, bt_z = make_zero(at, a), make_zero(bt, b)

        # # tan0 = mcbadd(a, bt_z) # TODO which one ? execute on device, or not? Transpose rule necessary for mcbadd then.
        # tan0 = a @ bt_z
        # # tan0 = a.conductences @ bt_z
        # tan1 = at_z @ b
        # output_tangent = tan0 + tan1 # TODO what should this be ? or with memristive matmul, at least the first one...
        
        # calculate gradient with respect to conductances
        if type(at) is not ad.Zero:
            tan0 = at + b
            calc_mat0_grad = True
        else:
            calc_mat0_grad = False

        # calculate gradient with respect to inputs
        if type(bt) is not ad.Zero:
            tan1 = a + bt
            calc_mat1_grad = True
        else:
            calc_mat1_grad = False

        # sum the gradients
        if calc_mat0_grad and calc_mat1_grad:
            output_tangent = tan0 + tan1
        elif calc_mat0_grad:
            output_tangent = tan0
        elif calc_mat1_grad:
            output_tangent = tan1
        
        return (primal_out, output_tangent)

    def _mcbadd_batch(vector_arg_values, batch_axes):
        """Computes the batched version of the primitive.
        
        This must be a JAX-traceable function.
        
        Since the multiply_add primitive already operates pointwise on arbitrary
        dimension tensors, to batch it we can use the primitive itself. This works as
        long as both the inputs have the same dimensions and are batched along the
        same axes. The result is batched along the axis that the inputs are batched.
        
        Args:
            vector_arg_values: a tuple of two arguments, each being a tensor of matching
            shape.
            batch_axes: the axes that are being batched. See vmap documentation.
        Returns:
            a tuple of the result, and the result axis that was batched. 
        """
        if batch_axes[0] is not None:
            raise ValueError("Batching over memristive crossbar array matrix maultiply with respect to matrix parameters is not implemented yet.")
        num_batches = vector_arg_values[1].shape[batch_axes[1]]
        res_list = []
        # a_batched = jnp.moveaxis(vector_arg_values[0], batch_axes[0], 0)
        a = vector_arg_values[0]
        b_batched = jnp.moveaxis(vector_arg_values[1], batch_axes[1], 0)
        for ibatch in range(num_batches):
            res_list.append(mcbadd(a, b_batched[ibatch]))
        # res = jnp.stack(res_list, axis=batch_axes[0])
        # return res, batch_axes[0]
        res = jnp.stack(res_list, axis=0)
        return res, 0

    def create_pycapsule(fn_prt):
        PyCapsule_Destructor = ctypes.CFUNCTYPE(None, ctypes.py_object)
        PyCapsule_New = ctypes.pythonapi.PyCapsule_New
        PyCapsule_New.restype = ctypes.py_object
        PyCapsule_New.argtypes = (ctypes.c_void_p, ctypes.c_char_p, PyCapsule_Destructor)
        capsule = PyCapsule_New(fn_prt, b"xla._CUSTOM_CALL_TARGET", PyCapsule_Destructor(0))
        return capsule

    # so_file = "./libfuncs.so"
    my_functions = ctypes.CDLL(so_file)
    cpu_add_f32_fn = my_functions.cpu_mcbadd_f32
    cpu_add_f32_fn_capsule = create_pycapsule(cpu_add_f32_fn)
    xla_client.register_custom_call_target(b"cpu_mcbadd_f32", cpu_add_f32_fn_capsule)

    # print(cpu_add_f32_fn)
    # print(type(cpu_add_f32_fn))
    # print(cpu_add_f32_fn_capsule)
    # print(type(cpu_add_f32_fn_capsule))

    _mcbadd_p = core.Primitive("cpu_mcbadd_f32")  # Create the primitive
    # _mcbadd_p.multiple_results = False # set for multi-output primitives.
    # _mcbadd_p.call_primitive = False # set for call primitives processed in final style.
    # _mcbadd_p.map_primitive = False # set for map primitives processed in final style.
    _mcbadd_p.def_impl(ft.partial(xla.apply_primitive, _mcbadd_p))
    _mcbadd_p.def_abstract_eval(_mcbadd_abstract_eval)
    xla.register_translation(_mcbadd_p, _mcbadd_xla_translation, platform='cpu')
    # ad.primitive_jvps[_mcbadd_p] = _mcbadd_value_and_jvp
    # batching.primitive_batchers[_mcbadd_p] = _mcbadd_batch


    return mcbadd

# if __name__ == '__main__':
#     # a = np.ones((4, 5), dtype=np.float32)
#     a = np.repeat(np.arange(4), 5).reshape((4, 5)).astype(np.float32)
#     b = np.arange(5, dtype=np.float32)

#     print(a)
#     print(b)

#     print("standard execution: ", mcbadd(a, b))
#     print("jit:                ", jax.jit(mcbadd)(a, b))
#     jvp_res = jax.jvp(mcbadd, (a, b), (np.ones_like(a), np.ones_like(b)))
#     print("jvp: primals:        ", jvp_res[0])
#     print("jvp: jvp:           ", jvp_res[1])
#     jvp_res_jit = jax.jit(lambda arg_values, arg_tangents: 
#                 jax.jvp(mcbadd, arg_values, arg_tangents))(
#                     (a, b), (np.ones_like(a), np.ones_like(b)))
#     print("jit jvp: output:    ", jvp_res_jit[0])
#     print("jit jvp: jvp:       ", jvp_res_jit[1])
#     # primals, f_vjp = jax.vjp(custom_add, a, b)
#     # jvp_res = f_vjp((np.ones(5), np.zeros(5)))
#     # print("vjp: primals:        ", jvp_res[0])
#     # print("vjp: vjp:           ", jvp_res[1])

#     # # NOTE : if jvp is defined in terms of jax known expressions, transpose can be found automatically
#     # # NOTE : if jvp is defined in terms of custom ops, transpose has to be defined manually for the custom op
#     # sum_val, grad_of_sum = jax.value_and_grad(lambda a, b: mcbadd(a, b).sum())(a, b)
#     # print("grad of sum:        ", grad_of_sum)

#     num_batches = 4
#     # a_batched = np.repeat(a[None, :], num_batches, axis=0)
#     # b_batched = np.repeat(b[:, None], num_batches, axis=1)
#     b_batched = np.arange(5*num_batches, dtype=np.float32).reshape((5, num_batches))
#     batched_res = jax.vmap(mcbadd, in_axes=(None,1), out_axes=0)(a, b_batched)
#     print("vmap: ", batched_res)

#     sum_val, grad_of_sum = jax.jit(jax.vmap(jax.value_and_grad(lambda a, b: mcbadd(a, b).sum(), argnums=(0, 1)), in_axes=(None,1), out_axes=0))(a, b_batched)
#     print("jit vmap grad:(prim)", sum_val)
#     print("jit vmap grad:(grad)", grad_of_sum)
#     sum_val, grad_of_sum = jax.jit(jax.vmap(jax.value_and_grad(lambda a, b: (a @ b).sum(), argnums=(0, 1)), in_axes=(None,1), out_axes=0))(a, b_batched)
#     print("\nreference")
#     print("jit vmap grad:(prim)", sum_val)
#     print("jit vmap grad:(grad)", grad_of_sum)



# # builder: jaxlib.xla_extension.XlaBuilder,                           call_target_name: bytes,            operands: Span[jaxlib.xla_extension.XlaOp],                                                                                                                                                                                                                                                                                                         shape_with_layout: jaxlib.xla_extension.Shape, operand_shapes_with_layout: Span[jaxlib.xla_extension.Shape], opaque: bytes = b'', has_side_effect: bool = False, schedule: jaxlib.xla_extension.ops.CustomCallSchedule = <CustomCallSchedule.SCHEDULE_NONE: 0>, api_version: jaxlib.xla_extension.ops.CustomCallApiVersion = <CustomCallApiVersion.API_VERSION_ORIGINAL: 1>) -> jaxlib.xla_extension.XlaOp
# # builder=<jaxlib.xla_extension.XlaBuilder object at 0x7f5e822f1c30>, call_target_name=b'cpu_mcbadd_f32', operands=(<jaxlib.xla_extension.XlaOp object at 0x7f5e822f1d70>, <jaxlib.xla_extension.XlaOp object at 0x7f5e822f1db0>, <jaxlib.xla_extension.XlaOp object at 0x7f5e822f1e30>, <jaxlib.xla_extension.XlaOp object at 0x7f5e822f1e70>, <jaxlib.xla_extension.XlaOp object at 0x7f5e822f1eb0>, <jaxlib.xla_extension.XlaOp object at 0x7f5e822f1d30>), shape_with_layout=f32[4]{0},                    operand_shapes_with_layout=((s64[],), (s64[],), f32[4,5]{1,0}, s64[4]{0}, s64[5]{0}, f32[5]{0})
