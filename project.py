from keras.datasets import mnist, cifar10
from keras.layers import Input, Dense
from keras import regularizers, models, optimizers
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# Analytical PCA of the training set
def AnalyticalPCA(y, dimension):
    pca = PCA(n_components=dimension)
    pca.fit(y)
    loadings = pca.components_
    return loadings


# Linear Autoencoder
def LinearAE(y, dimension, learning_rate=1e-4, regularization=5e-4, epochs=10):
    input = Input(shape=(y.shape[1],))
    encoded = Dense(
        dimension,
        activation="relu",
        kernel_regularizer=regularizers.l2(regularization),
    )(input)
    decoded = Dense(
        y.shape[1],
        activation="linear",
        kernel_regularizer=regularizers.l2(regularization),
    )(encoded)
    autoencoder = models.Model(input, decoded)
    autoencoder.compile(
        optimizer=optimizers.Adam(lr=learning_rate), loss="mean_squared_error"
    )
    autoencoder.fit(y, y, epochs=epochs, batch_size=4, shuffle=True)
    (w1, b1, w2, b2) = autoencoder.get_weights()
    return (w1, b1, w2, b2)


def PlotResults(p, dimension, name):
    sqrt_dimension = int(np.ceil(np.sqrt(dimension)))
    plt.figure()
    for i in range(p.shape[0]):
        plt.subplot(sqrt_dimension, sqrt_dimension, i + 1)
        plt.imshow(p[i, :, :], cmap="gray")
        plt.axis("off")
    plt.savefig(name + ".png")


def PlotColorResults(p, dimension, name):
    sqrt_dimension = int(np.ceil(np.sqrt(dimension)))
    plt.figure()
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(p[i, :, :, :] * 255)
        plt.axis("off")
    plt.savefig(name + ".png")


def run_mnist():
    dimension = 16  # feel free to change this, but you may have to tune hyperparameters
    (y, _), (_, _) = mnist.load_data()  # load MNIST training images
    shape_y = y.shape  # store shape of y before reshaping it
    y = (
        np.reshape(y, [shape_y[0], shape_y[1] * shape_y[2]]).astype("float32") / 255
    )  # reshape y to be a 2D matrix of the dataset
    p_analytical = AnalyticalPCA(y, dimension)  # PCA by applying SVD to y
    (_, _, w2, _) = LinearAE(y, dimension, epochs=10)  # train a linear autoencoder
    (p_linear_ae, _, _) = np.linalg.svd(
        w2.T, full_matrices=False
    )  # PCA by applying SVD to linear autoencoder weights
    p_analytical = np.reshape(
        p_analytical, [dimension, shape_y[1], shape_y[2]]
    )  # reshape loading vectors before plotting
    w2 = np.reshape(
        w2, [dimension, shape_y[1], shape_y[2]]
    )  # reshape autoencoder weights before plotting
    p_linear_ae = np.reshape(
        p_linear_ae.T, [dimension, shape_y[1], shape_y[2]]
    )  # reshape loading vectors before plotting
    PlotResults(p_analytical, dimension, "MNIST_AnalyticalPCA")
    PlotResults(w2, dimension, "MNIST_W2")
    PlotResults(p_linear_ae, dimension, "MNIST_LinearAE_PCA")


def run_cifar():
    dimension = 36  # feel free to change this, but you may have to tune hyperparameters
    (y, _), (_, _) = cifar10.load_data()  # load MNIST training images
    shape_y = y.shape  # store shape of y before reshaping it
    y = (
        np.reshape(y, [shape_y[0], shape_y[1] * shape_y[2] * shape_y[3]]).astype(
            "float32"
        )
        / 255
    )  # reshape y to be a 2D matrix of the dataset
    p_analytical = AnalyticalPCA(y, dimension)  # PCA by applying SVD to y
    (_, _, w2, _) = LinearAE(y, dimension, epochs=20)  # train a linear autoencoder
    (p_linear_ae, _, _) = np.linalg.svd(
        w2.T, full_matrices=False
    )  # PCA by applying SVD to linear autoencoder weights
    p_analytical = np.reshape(
        p_analytical, [dimension, shape_y[1], shape_y[2], shape_y[3]]
    )  # reshape loading vectors before plotting
    w2 = np.reshape(
        w2, [dimension, shape_y[1], shape_y[2], shape_y[3]]
    )  # reshape autoencoder weights before plotting
    p_linear_ae = np.reshape(
        p_linear_ae.T, [dimension, shape_y[1], shape_y[2], shape_y[3]]
    )  # reshape loading vectors before plotting
    PlotColorResults(p_analytical, dimension, "CIFAR_AnalyticalPCA")
    PlotColorResults(w2, dimension, "CIFAR_W2")
    PlotColorResults(p_linear_ae, dimension, "CIFAR_LinearAE_PCA")


if __name__ == "__main__":
    run_mnist()
    run_cifar()
