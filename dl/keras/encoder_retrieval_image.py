import numpy as np
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.datasets import mnist
# import matplotlib.pyplot as plt


(X_train, _), (X_test, _) = mnist.load_data()

X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

print(X_train.shape, X_test.shape)

X_train = np.reshape(X_train, (-1, 28, 28, 1))
X_test = np.reshape(X_test, (-1, 28, 28, 1))

print(X_train.shape, X_test.shape)

input_img = Input(shape=(28, 28, 1))
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same', name='encoder')(x)

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')


autoencoder.fit(X_train, X_train, epochs=2, batch_size=32, callbacks=None)

autoencoder.save('autoencoder.h5')
autoencoder.summary()

# create encoder pa).outputt

encoder = Model(input=autoencoder.input, outputs=autoencoder.get_layer('encoder').output)
encoder.save('encoder.h5')

# test
query = X_test[7]

print(str(X_test.shape))

X_test = np.delete(X_test, 7, axis=0)

print(str(X_test.shape))

codes = encoder.predict(X_test)
query_code = encoder.predict(query.reshape(1, 28, 28, 1))

from sklearn.neighbors import NearestNeighbors
n_neigh = 5

codes = codes.reshape(-1, 4 * 4 * 8)
print(codes.shape)
query_code = query_code.reshape(1, 4 * 4 * 8)
print(query_code.shape)

nbrs = NearestNeighbors(n_neighbors=n_neigh).fit(codes)

distances, indices = nbrs.kneighbors(np.array(query_code))

closest_images = X_test[indices]

closest_images = closest_images.reshape(-1, 28, 28, 1)
print(closest_images.shape)
