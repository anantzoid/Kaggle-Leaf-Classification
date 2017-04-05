from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
import matplotlib.pyplot as plt
import os
from scipy.misc import imread, imresize
import numpy as np

model = "m2"

img_height, img_width = 256, 256
input_img = Input(shape=(img_height, img_width,1))

x = Convolution2D(128, 5, 5, activation='relu', border_mode='same')(input_img)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(64, 5, 5, activation='relu', border_mode='same')(x)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(32, 5, 5, activation='relu', border_mode='same')(x)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(16, 5, 5, activation='relu', border_mode='same')(x)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(8, 5, 5, activation='relu', border_mode='same')(x)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(4, 5, 5, activation='relu', border_mode='same')(x)
encoded = MaxPooling2D((2, 2), border_mode='same')(x)

# at this point the representation is (8, 4, 4) i.e. 128-dimensional

x = Convolution2D(4, 5, 5, activation='relu', border_mode='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(8, 5, 5, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(16, 5, 5, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(32, 5, 5, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(64, 5, 5, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(128, 5, 5, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Convolution2D(1, 5, 5, activation='sigmoid', border_mode='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
if os.path.exists("%s_model_weights.h5"%model):
    autoencoder.load_weights("%s_model_weights.h5"%model)

image_data = [] 
for image_file in os.listdir('data/images'):
    image_data.append(imresize(imread(os.path.join('data/images', image_file)),(img_height,img_width)).reshape((img_height, img_width,1)).astype(np.float32))
image_data = np.array(image_data)
image_data = (np.random.random(image_data.shape) < image_data).astype(np.float32)
autoencoder.fit(image_data, image_data, nb_epoch=100, batch_size=128, shuffle=True)

autoencoder.save_weights("model_weights.h5")
decoded_imgs = autoencoder.predict(image_data[:20])

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(image_data[i].reshape(img_height, img_width), cmap='binary')
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n + 1)
    plt.imshow(decoded_imgs[i].reshape(img_height, img_width), cmap='binary')
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
