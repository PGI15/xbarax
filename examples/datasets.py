import numpy as np
import urllib.request
import gzip
import os

def load_mnist():
    base_target_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "datasets", "mnist")
    os.makedirs(base_target_path, exist_ok=True)
    base_url = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
    files = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
             't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
    paths_source = []
    paths_target = []
    for file in files:
        paths_source.append(base_url + file)
        paths_target.append(os.path.join(base_target_path, file))

    # Download the files if they don't exist
    for path_s, path_t in zip(paths_source, paths_target):
        if not os.path.exists(path_t):
            print("Downloading", os.path.basename(path_s), "...")
            urllib.request.urlretrieve(path_s, path_t)

    # Load the training data
    with gzip.open(paths_target[0], 'rb') as f:
        X_train = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28*28)
    with gzip.open(paths_target[1], 'rb') as f:
        y_train = np.frombuffer(f.read(), np.uint8, offset=8)

    # Load the test data
    with gzip.open(paths_target[2], 'rb') as f:
        X_test = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28*28)
    with gzip.open(paths_target[3], 'rb') as f:
        y_test = np.frombuffer(f.read(), np.uint8, offset=8)

    return (X_train, y_train), (X_test, y_test)


if __name__ == "__main__":
    # Load the dataset
    (X_train, y_train), (X_test, y_test) = load_mnist()

    # Normalize the input data
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # Print the shape of the dataset
    print("Training data shape:", X_train.shape)
    print("Training label shape:", y_train.shape)
    print("Test data shape:", X_test.shape)
    print("Test label shape:", y_test.shape)