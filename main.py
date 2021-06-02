from VaDE import VaDE
import numpy as np
import warnings
warnings.filterwarnings("ignore")


lr = 0.002
epochs = 300
batch_size = 100
n_centroid = 10


def loaddata():
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.reshape(x_train, newshape=(-1, 784))
    x_test = np.reshape(x_test, newshape=(-1, 784))
    x_train = np.concatenate((x_train, x_test), axis=0)
    y_train = np.concatenate((y_train, y_test), axis=0)
    shuffleindex = np.array(range(x_train.shape[0]))
    np.random.seed(0)
    np.random.shuffle(shuffleindex)
    x_train = x_train.astype('float32')[shuffleindex] / 255
    y_train = y_train[shuffleindex]
    x_test = x_test.astype('float32') / 255
    return [x_train, y_train], [x_test, y_test]


if __name__ == '__main__':
    train, test = loaddata()
    x_dim = train[0].shape[1]
    z_dim = 10
    encoder_layerinfo = [500, 500, 2000]
    decoder_layerinfo = [2000, 500, 500]
    for i in range(100):
        model = VaDE(
            encoder_layerinfo, decoder_layerinfo, lr,
            epochs, batch_size, x_dim, z_dim, n_centroid,
            train[0], test[0], train[1], test[1]
        )
        model.train()
