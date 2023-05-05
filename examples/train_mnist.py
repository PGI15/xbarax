import functools as ft
import jax
jax.config.update('jax_platform_name', 'cpu') # use cpu (xbar backend is only defined for cpu host)
import jax.numpy as jnp
from jax import jit
import jax.random as jrandom
import optax

# Train the network
USE_XBAR = True
LAYER_SIZES = [784, 128, 10]  # 2-layer MLP
LEARNING_RATE = 0.01
NUM_EPOCHS = 100
BATCH_SIZE = 128

# Load the MNIST dataset
from datasets import load_mnist
(X_train, y_train), (X_test, y_test) = load_mnist()

# Flatten the input data
X_train = X_train.reshape(X_train.shape[0], -1).astype(jnp.float32) / 255.0
X_test = X_test.reshape(X_test.shape[0], -1).astype(jnp.float32) / 255.0

# Define the network structure
def mlp(params, x):
    for w, b in params[:-1]:
        x = w @ x + b
        x = jax.nn.relu(x)  # ReLU activation
    x = params[-1][0] @ x
    return x

def init_params(LAYER_SIZES, key):
    keys = jrandom.split(key, len(LAYER_SIZES))
    return [[jrandom.normal(keyi, (out_size, in_size)), 
             jnp.zeros(out_size)]
            for keyi, (in_size, out_size) in zip(keys, 
                                                 zip(LAYER_SIZES[:-1], LAYER_SIZES[1:]))]

@ft.partial(jax.vmap, in_axes=(None, 0))
def predict_batch(params, x):
    logits = mlp(params, x)
    return logits

def calc_loss_single(params, x, labels):
    logits = mlp(params, x)
    return optax.softmax_cross_entropy_with_integer_labels(logits, labels)

def calc_loss_batch(params, x, labels):
    return jax.vmap(calc_loss_single, in_axes=(None, 0, 0))(params, x, labels).sum()

def accuracy(params, x, y):
    return jnp.mean(jnp.argmax(predict_batch(params, x), axis=-1) == y)

key = jrandom.PRNGKey(0)
params = init_params(LAYER_SIZES, key)


if USE_XBAR:
    # replace the last layer weight with a memristive weight
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath("")), "..", "xbarax"))

    from xbarax.xla_crossbar_interface_singleBuf.custom_xla_matmul import get_mcbmm_fn
    from xbarax.xla_crossbar_interface_singleBuf.custom_xla_add import get_mcbadd_fn
    from xbarax.xla_crossbar_interface_singleBuf.custom_xla_array import MemristiveCrossbarArray, AbstractMemristiveCrossbarArray, set_memristive_crossbar_array_device_put_handler

    so_filename = "../xbarax/xla_crossbar_interface_singleBuf/libfuncs.so"
    set_memristive_crossbar_array_device_put_handler(so_filename)
    AbstractMemristiveCrossbarArray.set_matmul_fn(get_mcbmm_fn(so_filename))
    AbstractMemristiveCrossbarArray.set_add_fn(get_mcbadd_fn(so_filename))

    memristive_weight = MemristiveCrossbarArray(params[-1][0].copy().astype(jnp.float32))
    params[-1][0] = memristive_weight

optim = optax.sgd(LEARNING_RATE)
opt_state = jax.jit(optim.init)(params)

@jit
def update(params, x, y, opt_state):
    loss, grads = jax.value_and_grad(calc_loss_batch)(params, x, y)
    updates, opt_state = optim.update(grads, opt_state)
    new_params = jax.tree_util.tree_map(lambda p, u: p + u, params, updates)
    return new_params, opt_state, loss

# @jit
def train_epoch(params, X, y, opt_state, key):
    perms = jrandom.permutation(key, len(X))
    loss_vals = []
    iter_ = 0
    for i in range(0, len(X), BATCH_SIZE):
        batch_idx = perms[i:i+BATCH_SIZE]
        batch_X, batch_y = X[batch_idx], y[batch_idx]
        new_params, opt_state, loss_val = update(params, batch_X, batch_y, opt_state)
        params = new_params
        loss_vals.append(loss_val)
        iter_ += 1
    return params, opt_state, jnp.asarray(loss_vals).mean()

for epoch in range(NUM_EPOCHS):
    key, batch_key = jrandom.split(key, 2)
    params, opt_state, mean_loss = train_epoch(params, X_train, y_train, opt_state, batch_key)
    train_acc = accuracy(params, X_train, y_train)
    test_acc = accuracy(params, X_test, y_test)
    print(f"Epoch {epoch+1} - mean loss: {mean_loss:.3f} - train accuracy: {train_acc:.3f} - test accuracy: {test_acc:.3f}")