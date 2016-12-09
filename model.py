from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import pandas as pd
from scipy.misc import imread, imresize
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Merge
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import model_from_json, Model
from keras.callbacks import TensorBoard
from keras.layers.normalization import BatchNormalization


img_height, img_width = 128, 128
data_dir = 'data/' 

tb_logdir = '/tmp/leaf/run1/'
if not os.path.exists(tb_logdir):
    os.makedirs(tb_logdir)

model1_weights = "data/model1_weights"
if img_height == 128:
    model_weights_file = "data/prod/m1/m1_model_weights.h5"
else:
    model_weights_file = "data/prod/m2/m2_model_weights.h5"


train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
image_data = {}
images_dir = os.path.join(data_dir, 'images')
for image_file in os.listdir(images_dir):
    image_data[image_file.split(".")[0]] = imresize(imread(os.path.join(images_dir, image_file)),(img_height,img_width)).reshape((img_height, img_width,1)).astype(np.float32)

def make_submit(preds):
    submission = pd.DataFrame(preds, columns=classes)
    submission.insert(0, 'id', test_ids)
    submission.reset_index()
    submission.to_csv('submit.csv', index=False)

def prepare_data(train,  test, image_data):
    le = LabelEncoder().fit(train.species)
    labels = le.transform(train.species)
    labels_cat = to_categorical(labels)
    classes = list(le.classes_)

    test_ids = test.id
    train_ids = train.id
    image_train = np.array([image_data[str(_)] for _ in train_ids])
    image_test = np.array([image_data[str(_)] for _ in test_ids])

    train = train.drop(['id', 'species'], axis=1)
    test = test.drop(['id'], axis=1)

    scaler = StandardScaler().fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)

    return train, labels_cat, classes, test_ids, test, np.concatenate((image_train, image_test)) 

def m1_encode_images(images):
    
    images = (np.random.random(images.shape) < images).astype(np.float32)
    '''
    json_file = open(ae_model_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    autoencoder = model_from_json(loaded_model_json)
    '''

    input_img = Input(shape=(img_height, img_width,1))
    x = Convolution2D(32, 5, 5, activation='relu', border_mode='same')(input_img)
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
    decoded = Convolution2D(1, 5, 5, activation='sigmoid', border_mode='same')(x)
    autoencoder = Model(input_img, decoded)
    autoencoder.load_weights(model_weights_file)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    encoder = Model(input_img, encoded)
    return encoder.predict(images, batch_size=128)

def m2_encode_images(images):
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
    autoencoder.load_weights(model_weights_file)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    encoder = Model(input_img, encoded)
    return encoder.predict(images, batch_size=128)


def model1():

    model = Sequential()
    model.add(Dense(1024, input_dim=448, init="glorot_normal"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1024, init="glorot_normal"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(512, init="glorot_normal"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dense(99, init="glorot_normal"))
    model.add(Activation("softmax"))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    if os.path.exists(model1_weights):
        model.load_weights(model1_weights)
    return model

'''
def model2(train, test, encoded_images):
    encoded_images = encoded_images.reshape(encoded_images.shape[0], -1)
    train = np.concatenate((train, encoded_images[:train.shape[0],:]), axis=1)
    test = np.concatenate((test, encoded_images[train.shape[0]:,:]), axis=1)
    
    img_model = Sequential()
    img_model.add(Dense(512, input_dim=256, init="glorot_normal", activation="relu"))
    
    model = Sequential()
    model.add(Merge([img_model, ]))
    model.add(Dense(1024, input_dim=448, init="glorot_normal"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1024, init="glorot_normal"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(512, init="glorot_normal"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dense(99, init="glorot_normal"))
    model.add(Activation("softmax"))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

'''

train, labels, classes, test_ids, test, images = prepare_data(train, test, image_data)
encoded_images = m1_encode_images(images) if img_height == 128 else m2_encode_images(images)
encoded_images = encoded_images.reshape(encoded_images.shape[0], -1)
train = np.concatenate((train, encoded_images[:train.shape[0],:]), axis=1)
test = np.concatenate((test, encoded_images[train.shape[0]:,:]), axis=1)

model = model1()
#val_split = train.shape[0] // 9
model.fit(train, labels, nb_epoch=10, batch_size=256)
model.save_weights(model1_weights)

preds = model.predict_proba(test)
make_submit(preds)


