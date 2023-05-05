# xbarax
A simple Jax and XLA interface for memristive crossbar arrays.

## First steps

Clone the repository, install the requirements and compile the interface. You might want to create a virtual environment first. 

```console
git clone git@github.com:PGI15/xbarax.git
pip install -r xbarax/requirements.txt
cd xbarax/xbarax/xla_crossbar_interface_singleBuf/
gcc -shared -fPIC -o libfuncs.so xla_interface.c user_implementation.h
```

And then run any of the examples in the `examples` folder:
- `first_steps.py` demonstrates how to use the `MemrsitiveCrossbarArray` class
- `jaxpr_comparison.py` compares the `jaxpr` generated by standard jax/numpy arrays and the `MemrsitiveCrossbarArray`
- `train_nmnist.py` trains the mnist dataset with a MLP where the last weight is replaced by the crossbar array
- `snn_fc_jax_pure_crossbar.ipynb` integrates the crossbar array into a spiking neural network

Note, the examples might need additional requirements specified in `additional_requirements_examples.txt`.

## Interfacing your own crossbar array

Simply reimplement the functions
- `user_implementation_mul` for matrix multiplication
- `user_implementation_add` for adding values to the crossbar conductances
- `user_implementation_write` for writing an array of values to the crossbar array

by calling your own functions that executes the operations on the crossbar array within.
Recompile the code using `gcc -shared -fPIC -o libfuncs.so xla_interface.c user_implementation.h` and you should be able to use your crossbar via the `MemristiveCrossbarArray` class in your own jax usecases (e.g. the example scripts).

## Development

We will keep updating the package for it to work with the latest jax version and we will improve documentation.

## Citation
