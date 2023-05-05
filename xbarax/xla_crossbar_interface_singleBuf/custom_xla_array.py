import sys
from typing import Callable, Optional, Sequence
from dataclasses import dataclass
import numpy as np
import jax

from jax import jit, lax, make_jaxpr
from jax._src import device_array, core
from jax._src import dispatch
from jax._src import dtypes
from jax._src import ad_util
from jax.interpreters import mlir
from jax.interpreters import xla
from jax._src.lib.mlir import ir
from jax._src.lib import xla_client
from jax._src import xla_bridge
xc = xla_client
xb = xla_bridge

from jax.config import config
config.parse_flags_with_absl()

# TODO(jakevdp): use a setup/teardown method to populate and unpopulate all the
# dictionaries associated with the following objects.

# Define a sparse array data structure. The important feature here is that
# it is a jaxpr object that is backed by two device buffers.
class MemristiveCrossbarArray:
    """Data structure representing memristive crossbar array."""
    def __init__(self, conductences, aval=None, use_crossbar=True):
        # TODO corretly handle index_dtypes...
        assert conductences.dtype == np.float32, "Only float32 is supported for now"
        if aval is None:
            aval = AbstractMemristiveCrossbarArray(conductences.shape, conductences.dtype, use_crossbar=use_crossbar)
        self.aval = aval
        self.conductences = conductences

    @property
    def shape(self):
        return self.aval.shape

    @property
    def dtype(self):
        return self.aval.dtype

    @property
    def ndim(self):
        return self.aval.ndim

    @property
    def is_set(self):
        return self.was_set and (not self.is_deallocated) # and (self.use_crossbar) # TODO use this here as additional flag? _is_set might be True due to result handler for simulated arrays...

    @property
    def use_crossbar(self):
        return self.aval._use_crossbar

    @use_crossbar.setter
    def use_crossbar(self, use_crossbar_):
        if self.is_set:
            self.aval = self.aval.update(use_crossbar=use_crossbar_)
        else:
            self.aval = self.aval.update(use_crossbar=use_crossbar_, was_set=False, is_deallocated=False)

    @property
    def was_set(self):
        return self.aval.was_set

    @property
    def is_deallocated(self):
        return self.aval.is_deallocated

    def __matmul__(self, b):
        return self.aval._matmul(self, b)

    def __add__(self, b):
        res = self.aval._add(self, b)
        return res

    def __repr__(self):
        return f"MemristiveCrossbarArray({self.conductences}, shape={self.conductences.shape}, aval={self.aval.conductences_aval}, is_set={self.is_set}, use_crossbar={self.use_crossbar})"

conductences_p = core.Primitive('conductences')

@conductences_p.def_impl
def _conductences_impl(mat):
  return mat.conductences

@conductences_p.def_abstract_eval
def _conductences_abstract_eval(mat):
  return mat.conductences_aval

# Note: cannot use lower_fun to define attribute access primitives
# because it leads to infinite recursion.

def _conductences_mhlo_lowering(ctx, conductences):
  return (conductences, )

mlir.register_lowering(conductences_p, _conductences_mhlo_lowering)

class AbstractMemristiveCrossbarArray(core.ShapedArray):
    __slots__ = ['conductences_aval', '_use_crossbar', '_was_set', '_is_deallocated']

    _matmul_fn = None
    _add_fn = None
    # _use_crossbar = None

    def __init__(self, shape, dtype, weak_type=False,
                named_shape=None, use_crossbar=True, was_set=False, is_deallocated=False):
        super().__init__(shape, dtypes.canonicalize_dtype(dtype))
        assert len(shape) == 2
        named_shape = {} if named_shape is None else named_shape
        self.conductences_aval = core.ShapedArray(shape, dtypes.canonicalize_dtype(dtype),
                                        weak_type, named_shape)
        self._use_crossbar = use_crossbar
        self._is_deallocated = is_deallocated
        self._was_set = was_set

    def update(self, shape=None, dtype=None, 
                weak_type=None, named_shape=None, use_crossbar=None, was_set=None, is_deallocated=None):
        if shape is None:
            shape = self.shape
        if dtype is None:
            dtype = self.dtype
        if weak_type is None:
            weak_type = self.weak_type
        if named_shape is None:
            named_shape = self.named_shape
        if use_crossbar is None:
            use_crossbar = self.use_crossbar
        if was_set is None:
            was_set = self.was_set
        if is_deallocated is None:
            is_deallocated = self.is_deallocated
        return type(self)(
            shape, dtype, weak_type, named_shape, use_crossbar, was_set, is_deallocated)

    def strip_weak_type(self):
        return self

    @core.aval_property
    def conductences(self):
        return conductences_p.bind(self)

    @property
    def use_crossbar(self):
        return self._use_crossbar

    @property
    def was_set(self):
        return self._was_set

    @property
    def is_deallocated(self):
        return self._is_deallocated

    def _check_status(self):
        if self.is_deallocated and self.use_crossbar:
            raise RuntimeError("`MemristiveCrossbarArray` is deallocated or overwritten from another instance on the crossbar device, e.g. due to an `add` operation.") # resulting in a new instance being allocated and this one being deallocated.")

    def _deallocate(self):
        self._is_deallocated = True

    @classmethod
    def set_matmul_fn(cls, matmul_fn: Callable):
        cls._matmul_fn = staticmethod(matmul_fn)

    def _matmul(self, a, b):
        self._check_status()
        if not self.use_crossbar:
            raise ValueError("`use_corssbar=False` is not properly implemented yet.")
        if self.use_crossbar:
            res = self._matmul_fn(a, b)  
        else:
            res = super()._matmul(a.conductences, b) # TODO not necessarily first argument is the memristive crossbar array
        return res

    @classmethod
    def set_rmatmul_fn(cls, rmatmul_fn: Callable):
        cls._rmatmul_fn = staticmethod(rmatmul_fn)

    def _rmatmul(self, a, b):
        return self._rmatmul_fn(a, b) 

    # @classmethod
    # def set_add_fn(cls, add_fn: Callable):
    #     cls._add = staticmethod(add_fn)

    @classmethod
    def set_add_fn(cls, add_fn: Callable):
        cls._add_fn = staticmethod(add_fn)
    
    def _add(self, a, b):
        self._check_status()
        if self.use_crossbar:
            res = self._add_fn(a, b)
            # self._deallocate()
            # print(res)
            # res._is_deallocated = False
            # res._is_set = True
        else:
            res = super()._add(a.conductences, b)
        return res

def memristive_crossbar_array_result_handler(device, aval):
    # def build_memristive_crossbar_array(_, conductences_buf):
    def build_memristive_crossbar_array(_, conductences_buf):
        conductences = device_array.make_device_array(aval.conductences_aval, device, conductences_buf)
        aval_upd = aval.update(is_deallocated=False, was_set=True) # TODO check whether even necessary...
        return MemristiveCrossbarArray(conductences, aval_upd) # `is_set` as the only way device array is created as a result is from a add operation, meaning from a presiously set array is `use_crossbar is True`
    return build_memristive_crossbar_array

def memristive_crossbar_array_shape_handler(a):
    # print("memristive_crossbar_array_shape_handler")
    return (xc.Shape.array_shape(a.conductences_aval.dtype, a.conductences_aval.shape), )

# def set_memristive_crossbar_array_device_put_handler_python(device_set_fn=None):
#     def memristive_crossbar_array_device_put_handler(a, device):
#         if (device_set_fn is not None) and (not a.is_set) and (a.use_crossbar):
#             device_set_fn(a)
#             a._is_set = True
#         return (*dispatch.device_put(a.conductences, device), )
#     dispatch.device_put_handlers[MemristiveCrossbarArray] = memristive_crossbar_array_device_put_handler
import ctypes

def set_memristive_crossbar_array_device_put_handler(so_file: str, fn_name: str = None):
    if fn_name is None:
        fn_name = "user_implementation_write"
    my_functions = ctypes.CDLL(so_file)
    write_fn = getattr(my_functions, fn_name)

    def memristive_crossbar_array_device_put_handler(a, device):
        if (not a.is_set) and (a.use_crossbar):
            conds_ctype = np.ravel(np.asarray(a.conductences)).ctypes.data_as(ctypes.POINTER(ctypes.c_void_p))
            write_fn(conds_ctype, a.conductences.shape[0], a.conductences.shape[1])
            a.aval = a.aval.update(was_set=True, is_deallocated=False)
        return (*dispatch.device_put(a.conductences, device), )
    dispatch.device_put_handlers[MemristiveCrossbarArray] = memristive_crossbar_array_device_put_handler

core.pytype_aval_mappings[MemristiveCrossbarArray] = lambda x: x.aval
core.raise_to_shaped_mappings[AbstractMemristiveCrossbarArray] = lambda aval, _: aval
xla.pytype_aval_mappings[MemristiveCrossbarArray] = lambda x: x.aval
xla.canonicalize_dtype_handlers[MemristiveCrossbarArray] = lambda x: x
# # dispatch.device_put_handlers[MemristiveCrossbarArray] = memristicve_crossbar_array_device_put_handler
# set_memristive_crossbar_array_device_put_handler_python()
dispatch.result_handlers[AbstractMemristiveCrossbarArray] = memristive_crossbar_array_result_handler
dispatch.num_buffers_handlers[AbstractMemristiveCrossbarArray] = lambda _: 1
xla.xla_shape_handlers[AbstractMemristiveCrossbarArray] = memristive_crossbar_array_shape_handler

def memristive_crossbar_array_mlir_type_handler(a):
  return (ir.RankedTensorType.get(
          a.conductences_aval.shape, mlir.dtype_to_ir_type(a.conductences_aval.dtype)), )

mlir.ir_type_handlers[AbstractMemristiveCrossbarArray] = memristive_crossbar_array_mlir_type_handler # TODO change this guy for single array handling in add ?


from jax._src.lax.lax import _convert_element_type, broadcast
def zeros_like_memristive_crossbar_array(aval):
    assert isinstance(aval, AbstractMemristiveCrossbarArray)
    if aval.dtype == dtypes.float0:
        scalar_zero = np.zeros((), dtype=aval.dtype)
    else:
        scalar_zero = _convert_element_type(0, aval.dtype, aval.weak_type)
    # return type(aval)(broadcast(scalar_zero, aval.shape))
    return broadcast(scalar_zero, aval.shape)

ad_util.aval_zeros_likers[AbstractMemristiveCrossbarArray] = zeros_like_memristive_crossbar_array



# necessary for constant type handler (e.g. see zeros above)
# TODO works for now... in the future more sophisticated implementation might be necessary (see code above)
def _memristive_crossbar_array_constant_handler(val: MemristiveCrossbarArray, canonicalize_types
                             ) -> Sequence[ir.Value]:
    return mlir.get_constant_handler(type(val.conductences))(val.conductences, canonicalize_types)

mlir.register_constant_handler(MemristiveCrossbarArray, _memristive_crossbar_array_constant_handler)



from jax._src.typing import ArrayLike
from jax.interpreters import pxla, xla, mlir
from jax._src.sharding import (
    Sharding, SingleDeviceSharding, XLACompatibleSharding, PmapSharding,
    device_replica_id_map)
from jax._src.array import ArrayImpl, _array_shard_arg, _array_global_result_handler, _array_local_result_handler


def _memristive_crossbar_array_shard_arg(x, devices, indices, mode):
    if len(devices) > 1:
        raise NotImplementedError("Sharding `MemristiveCrossbarArray` to multiple devices is not supported.")
    # # TODO really device_put? will also put the conductences on the CPU device and not just call the memristor set function
    # x.aval = x.aval.update(is_deallocated = False)
    # print("shard", x)
    # print("shard", x.is_set, x.was_set, x.is_deallocated)
    if not x.was_set:
        dispatch.device_put(x, devices[0])
    conductences_shard = pxla.shard_arg_handlers[type(x.conductences)](x.conductences, devices, indices, mode)
    return conductences_shard

pxla.shard_arg_handlers[MemristiveCrossbarArray] = _memristive_crossbar_array_shard_arg



# def _array_global_result_handler(global_aval, out_sharding, committed,
#                                  is_out_sharding_from_xla):
#   if global_aval.dtype == dtypes.float0:
#     return lambda _: np.zeros(global_aval.shape, dtypes.float0)  # type: ignore
#   if core.is_opaque_dtype(global_aval.dtype):
#     return global_aval.dtype._rules.global_sharded_result_handler(
#         global_aval, out_sharding, committed, is_out_sharding_from_xla)
#   return lambda bufs: ArrayImpl(global_aval, out_sharding, bufs,
#                                 committed=committed, _skip_checks=True)


# def _array_global_result_handler_here(global_aval, out_sharding, committed,
#                                  is_out_sharding_from_xla):
#   print("_array_global_result_handler")
#   print("global_aval", global_aval)
#   if global_aval.dtype == dtypes.float0:
#     return lambda _: np.zeros(global_aval.shape, dtypes.float0)  # type: ignore
#   if core.is_opaque_dtype(global_aval.dtype):
#     return global_aval.dtype._rules.global_sharded_result_handler(
#         global_aval, out_sharding, committed, is_out_sharding_from_xla)
# #   print("xla_extension_version", xla_extension_version)
# #   if xla_extension_version >= 131:
# #     return xc.array_result_handler(
# #         global_aval, out_sharding, committed=committed, _skip_checks=True
# #     )
#   return lambda bufs: ArrayImpl(global_aval, out_sharding, bufs,
#                                 committed=committed, _skip_checks=True)



def _memristive_crossbar_array_global_result_handler(global_aval, out_sharding, committed,
                                 is_out_sharding_from_xla):
    # print("_memristive_crossbar_array_global_result_handler")
    # print(out_sharding, committed, is_out_sharding_from_xla)
    # sys.exit()
    array_result_handler = _array_global_result_handler(global_aval.conductences_aval, out_sharding, committed,
                                 is_out_sharding_from_xla)
    # if global_aval.is_deallocated: # TODO check whether even necessary...
    global_aval = global_aval.update(is_deallocated=False, was_set=True) 
    def _global_result_handler(bufs):
        array_impl = array_result_handler(bufs)
        return MemristiveCrossbarArray(array_impl, aval=global_aval)
    return _global_result_handler

# def _array_global_result_handler(global_aval, out_sharding, committed,
#                                  is_out_sharding_from_xla):
#   if global_aval.dtype == dtypes.float0:
#     return lambda _: np.zeros(global_aval.shape, dtypes.float0)  # type: ignore
#   if core.is_opaque_dtype(global_aval.dtype):
#     return global_aval.dtype._rules.global_sharded_result_handler(
#         global_aval, out_sharding, committed, is_out_sharding_from_xla)
#   return lambda bufs: ArrayImpl(global_aval, out_sharding, bufs,
#                                 committed=committed, _skip_checks=True)


#   if global_aval.dtype == dtypes.float0:
#     return lambda _: np.zeros(global_aval.shape, dtypes.float0)  # type: ignore
#   if core.is_opaque_dtype(global_aval.dtype):
#     return global_aval.dtype._rules.global_sharded_result_handler(
#         global_aval, out_sharding, committed, is_out_sharding_from_xla)
#   return lambda bufs: ArrayImpl(global_aval, out_sharding, bufs,
#                                 committed=committed, _skip_checks=True)


pxla.global_result_handlers[(AbstractMemristiveCrossbarArray, pxla.OutputType.Array)] = _memristive_crossbar_array_global_result_handler
# pxla.global_result_handlers[(core.ConcreteArray, pxla.OutputType.Array)] = _array_global_result_handler
# pxla.global_result_handlers[(core.AbstractToken, pxla.OutputType.Array)] = lambda *_: lambda *_: core.token


# Only used for Arrays that come out of pmap/sharding.
def _array_local_result_handler(aval, sharding, indices):
  if core.is_opaque_dtype(aval.dtype):
    return aval.dtype._rules.local_sharded_result_handler(
        aval, sharding, indices)
  return lambda bufs: ArrayImpl(aval, sharding, bufs, committed=True,
                                _skip_checks=True)

# _array_local_result_handler

def _memristive_crossbar_array_local_result_handler(aval, sharding, indices):
    array_result_handler = _array_local_result_handler(aval.conductences_aval, sharding, indices)
    if aval.is_deallocated: # TODO check whether even necessary...
        aval = aval.update(is_deallocated=False)
    def _local_result_handler(bufs):
        array_impl = array_result_handler(bufs)
        return MemristiveCrossbarArray(array_impl, aval=aval, is_set=True)
    return _local_result_handler


# pxla.local_result_handlers[(core.ShapedArray, pxla.OutputType.Array)] = _memristive_crossbar_array_local_result_handler
# pxla.local_result_handlers[(core.ShapedArray, pxla.OutputType.Array)] = _memristive_crossbar_array_local_result_handler
pxla.local_result_handlers[(AbstractMemristiveCrossbarArray, pxla.OutputType.Array)] = _memristive_crossbar_array_local_result_handler
# pxla.local_result_handlers[(core.ConcreteArray, pxla.OutputType.Array)] = _array_local_result_handler




