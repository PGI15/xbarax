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

# Helpful links:
# https://jax.readthedocs.io/en/latest/notebooks/How_JAX_primitives_work.html
# https://dfm.io/posts/extending-jax/
# https://github.com/dfm/extending-jax/blob/main/lib/cpu_ops.cc

def get_mcbmm_fn(so_file):


    # @jax.custom_vjp
    # def mcbmm(a, b):
    #     return mcbmm_fn(a, b)

    # def f_fwd(x, y):
    #     # Returns primal output and residuals to be used in backward pass by f_bwd.
    #     return f(x, y), (jnp.cos(x), jnp.sin(x), y)

    # def f_bwd(res, g):
    #     cos_x, sin_x, y = res # Gets residuals computed in f_fwd
    #     return (cos_x * g * y, sin_x * g)

    # mcbmm.defvjp(f_fwd, f_bwd)


    # @jax.custom_vjp
    def mcbmm(a, b):
        """The JAX-traceable way to use the JAX primitive.
        
        Note that the traced arguments must be passed as positional arguments
        to `bind`. 
        """
        #   return _mcbmm_p.bind(len(a), a, b)
        return _mcbmm_p.bind(a, b)

    # def mcbmm_fwd(a, b):
    #     # Returns primal output and residuals to be used in backward pass by f_bwd.
    #     return mcbmm(a, b), (a, b)

    # def mcbmm_bwd(res, g):
    #     (a, b) = res # Gets residuals computed in f_fwd
    #     # a.conductences to perform bwd pass on host
    #     return (jnp.outer(g, b), g @ a.conductences)

    # mcbmm.defvjp(mcbmm_fwd, mcbmm_bwd)


    def _mcbmm_abstract_eval(a, b):
        assert len(a.shape) == 2
        assert len(b.shape) == 1
        assert a.shape[-1] == b.shape[0]
        assert a.dtype == np.dtype("float32")
        assert b.dtype == np.dtype("float32")
        c_shape = (*a.shape[:-1], *b.shape[1:])
        return abstract_arrays.ShapedArray(c_shape, np.dtype("float32"))

    def _mcbmm_xla_translation(ctx, avals_in, avals_out, ac, bc):
    # def _mcbmm_xla_translation(ctx, avals_in, avals_out, ac, bc):
        # The inputs have "shapes" that provide both the shape and the dtype

        # print(ac)
        # print(bc)
        # ac.shape_tuples()
        # sys.exit()

        # condc, rowidc, colidc = ac

        ac_shape = avals_in[0]
        bc_shape = avals_in[1]
        cc_shape = avals_out[0]

        # print(avals_in)
        # print(avals_out)
        # sys.exit()
        # Extract the dtype and shape
        dtype = ac_shape.dtype
        dims = ac_shape.shape
        assert len(dims) == 2
        assert bc_shape.dtype == dtype
        assert bc_shape.shape[0] == dims[-1]
        assert len(bc_shape.shape) == 1
        assert cc_shape.dtype == dtype
        assert cc_shape.shape == (*ac_shape.shape[:-1], *bc_shape.shape[1:])

        # # The total size of the input is the product across dimensions
        # dim0 = xla_client.ops.ConstantLiteral(ctx, np.asarray(ac_shape.shape[0]).sum().astype(np.int64))
        # dim1 = xla_client.ops.ConstantLiteral(ctx, np.asarray(ac_shape.shape[1]).sum().astype(np.int64))

        # print(condc)
        # print(dim0)
        # sys.exit()

        # val = xla_client.LiteralSlice(4)
        # sys.exit()

        # dim0 = xla_client.ops.ConstantLiteral(ctx.builder, np.asarray(ac_shape.shape[0]).sum().astype(np.int64))
        # dim1 = xla_client.ops.ConstantLiteral(ctx.builder, np.asarray(ac_shape.shape[1]).sum().astype(np.int64))
        dims = xla_client.ops.ConstantLiteral(ctx.builder, np.asarray(ac_shape.shape, dtype=np.uint64))



        # # The inputs and outputs all have the same shape so let's predefine this
        # # specification
        # shape_dim0 = xla_client.Shape.array_shape(np.dtype(np.int64), (), ())
        # shape_dim1 = xla_client.Shape.array_shape(np.dtype(np.int64), (), ())
        shape_dims = xla_client.Shape.array_shape(np.dtype(np.uint64), (2,), (0,))

        shape_a = xla_client.Shape.array_shape(
            np.dtype(ac_shape.dtype), ac_shape.shape, tuple(range(len(ac_shape.shape) - 1, -1, -1))
        )
        shape_b = xla_client.Shape.array_shape(
            np.dtype(bc_shape.dtype), bc_shape.shape, tuple(range(len(bc_shape.shape) - 1, -1, -1))
        )
        shape_c = xla_client.Shape.array_shape(
            np.dtype(cc_shape.dtype), cc_shape.shape, tuple(range(len(cc_shape.shape) - 1, -1, -1))
        )

        # We dispatch a different call depending on the dtype
        if dtype == np.float32:
            op_name = b"cpu_mcbmm_f32"
        # elif dtype == np.float64:
        #     op_name = platform.encode() + b"_kepler_f64"
        else:
            raise NotImplementedError(f"Unsupported dtype {dtype}")

        # We pass the size of the data as a the first input argument
        return [xla_client.ops.CustomCallWithLayout(
            builder=ctx.builder,
            call_target_name=op_name,
            # operands=(dims, condc, rowidc, colidc, bc),
            operands=(dims, ac, bc),
            shape_with_layout=shape_c,
            operand_shapes_with_layout=(
                # shape_dim0,
                # shape_dim1,

                shape_dims,
                # xla_client.Shape.tuple_shape(
                shape_a,
                shape_b,
            ),
        ), ]


    def _mcbmm_value_and_jvp(arg_values, arg_tangents):
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
        primal_out = mcbmm(a, b)
        
        #   # We must use a JAX-traceable way to compute the tangent. It turns out that 
        #   # the output tangent can be computed as (xt * y + x * yt + zt),
        #   # which we can implement in a JAX-traceable way using the same "multiply_add_prim" primitive.
        
        # We do need to deal specially with Zero. Here we just turn it into a 
        # proper tensor of 0s (of the same shape as 'x'). 
        # An alternative would be to check for Zero and perform algebraic 
        # simplification of the output tangent computation.
        def make_zero(tan, likearr):
            return jax.lax.zeros_like_array(likearr) if type(tan) is ad.Zero else tan  
    
        # at_z, bt_z = make_zero(at, a), make_zero(bt, b)

        # # tan0 = mcbmm(a, bt_z) # TODO which one ? execute on device, or not? Transpose rule necessary for mcbmm then.
        # tan0 = a @ bt_z
        # # tan0 = a.conductences @ bt_z
        # tan1 = at_z @ b
        # output_tangent = tan0 + tan1 # TODO what should this be ? or with memristive matmul, at least the first one...


        # calculate gradient with respect to conductances
        if type(at) is not ad.Zero:
            tan0 = at @ b
            calc_mat_grad = True
        else:
            calc_mat_grad = False

        # calculate gradient with respect to inputs
        if type(bt) is not ad.Zero:
            tan1 = a @ bt
            calc_inp_grad = True
        else:
            calc_inp_grad = False

        # sum the gradients
        if calc_mat_grad and calc_inp_grad:
            output_tangent = tan0 + tan1
        elif calc_mat_grad:
            output_tangent = tan0
        elif calc_inp_grad:
            output_tangent = tan1


        # # tan0 = a @ bt_z
        # tan1 = at_z @ b        
        # output_tangent = tan1


        # print("_mcbmm_value_and_jvp")
        return (primal_out, output_tangent)

    def _mcbmm_transpose(ct, a, b):
        """Evaluates the transpose of a linear primitive.

        This method is only used when computing the backward gradient following 
        value_and_jvp, and is only needed for primitives that are used in the JVP 
        calculation for some other primitive. We need transposition for multiply_add_prim, 
        because we have used multiply_add_prim in the computation of the output_tangent in 
        multiply_add_value_and_jvp.

        In our case, multiply_add is not a linear primitive. However, it is used linearly 
        w.r.t. tangents in multiply_add_value_and_jvp:
            output_tangent(xt, yt, zt) = multiply_add_prim(xt, y, multiply_add_prim(x, yt, zt))
        
        Always one of the first two multiplicative arguments is a constant.

        Args:
            ct: the cotangent of the output of the primitive.
            x, y, z: values of the arguments. The arguments that are used linearly
                get an ad.UndefinedPrimal value. The other arguments get a constant
                value.
        Returns:
            a tuple with the cotangent of the inputs, with the value None
            corresponding to the constant arguments.
        """

        if not ad.is_undefined_primal(a):
            # This use of multiply_add is with a constant "x"
            assert ad.is_undefined_primal(b)
            # ct_b = ad.Zero(b.aval) if type(ct) is ad.Zero else mcbmm(ct, a) # TODO implement _rmatmul !!! with transpose flag
            ct_b = ad.Zero(b.aval) if type(ct) is ad.Zero else ct @ a.conductences # This run backward pass on host, not on device
            res = None, ct_b
        else:
            # This use of multiply_add is with a constant "y"
            assert ad.is_undefined_primal(a)
            ct_x = ad.Zero(b.aval) if type(ct) is ad.Zero else jnp.outer(ct, b) # This runs on host, not on device
            res = ct_x, None
        return res

    def _mcbmm_batch(vector_arg_values, batch_axes):
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
            raise ValueError("Batching over memristive crossbar array matrix multiply with respect to matrix parameters is not implemented yet.")
        num_batches = vector_arg_values[1].shape[batch_axes[1]]
        res_list = []
        # a_batched = jnp.moveaxis(vector_arg_values[0], batch_axes[0], 0)
        a = vector_arg_values[0]
        b_batched = jnp.moveaxis(vector_arg_values[1], batch_axes[1], 0)
        for ibatch in range(num_batches):
            res_list.append(mcbmm(a, b_batched[ibatch]))
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
    cpu_mcbmm_f32_fn = my_functions.cpu_mcbmm_f32
    cpu_mcbmm_f32_fn_capsule = create_pycapsule(cpu_mcbmm_f32_fn)
    xla_client.register_custom_call_target(b"cpu_mcbmm_f32", cpu_mcbmm_f32_fn_capsule)

    # print(cpu_add_f32_fn)
    # print(type(cpu_add_f32_fn))
    # print(cpu_add_f32_fn_capsule)
    # print(type(cpu_add_f32_fn_capsule))



    _mcbmm_p = core.Primitive("cpu_mcbmm_f32")  # Create the primitive
    _mcbmm_p.def_impl(ft.partial(xla.apply_primitive, _mcbmm_p))
    _mcbmm_p.def_abstract_eval(_mcbmm_abstract_eval)
    xla.register_translation(_mcbmm_p, _mcbmm_xla_translation, platform='cpu')
    ad.primitive_jvps[_mcbmm_p] = _mcbmm_value_and_jvp
    ad.primitive_transposes[_mcbmm_p] = _mcbmm_transpose
    batching.primitive_batchers[_mcbmm_p] = _mcbmm_batch

    return mcbmm