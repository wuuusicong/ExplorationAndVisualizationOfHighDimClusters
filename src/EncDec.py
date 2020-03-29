from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras.optimizers import RMSprop
import numpy as np

# this is the size of our encoded representations
encoding_dim = 4  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(784,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(784, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

#autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.compile(loss='mean_squared_error', optimizer=RMSprop())

is_fashion_mnist = False
if not is_fashion_mnist:
    (x_train, _), (x_test, y_test) = mnist.load_data()
else:
    (x_train, _), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print (x_train.shape)
print (x_test.shape)
if not is_fashion_mnist:
    np.save(file=f'../data/MNIST_AE/mnist_orig_dim', arr=x_test)
else:
    np.save(file=f'../data/FASHION_MNIST_AE/fashion_mnist_orig_dim', arr=x_test)

autoencoder.fit(x_train, x_train,
                # epochs=200,
                epochs=6,
                # batch_size=256,
                batch_size=32,
                shuffle=False,
                validation_data=(x_test, x_test),
                verbose=1)

# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test)
print('Saving dataset')
if not is_fashion_mnist:
    np.save(file=f'../data/MNIST_AE/mnist_encoded_to_{encoding_dim}_dim', arr=encoded_imgs)
    np.save(file=f'../data/MNIST_AE/mnist_labels_ae', arr=y_test)
else:
    np.save(file=f'../data/FASHION_MNIST_AE/fashion_mnist_encoded_to_{encoding_dim}_dim', arr=encoded_imgs)
    np.save(file=f'../data/FASHION_MNIST_AE/fashion_mnist_labels_ae', arr=y_test)
decoded_imgs = decoder.predict(encoded_imgs)

# use Matplotlib (don't ask)
import matplotlib.pyplot as plt

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()