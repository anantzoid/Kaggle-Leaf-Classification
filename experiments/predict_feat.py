# Score of 0.69394 on Kaggle. Rank 197
# Score of 0.03039 on Kaggle. Rank 147
# Score of 0.02026 on Kaggle. Rank 112
# Score of 0.03562 on Kaggle. Rank 102

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.misc import imread, imresize
import os
import matplotlib.pyplot as ply
from keras.utils.np_utils import to_categorical

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
img_height, img_width = 256, 256

def prepare_data(train,  test):
    le = LabelEncoder().fit(train.species)
    labels = le.transform(train.species)
    labels_cat = to_categorical(labels)
    classes = list(le.classes_)

    test_ids = test.id
    train_ids = train.id

    train = train.drop(['id', 'species'], axis=1)
    test = test.drop(['id'], axis=1)

    scaler = StandardScaler().fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)

    return train, labels_cat, classes, test_ids, test

def make_submit(preds):
    submission = pd.DataFrame(preds, columns=classes)
    submission.insert(0, 'id', test_ids)
    submission.reset_index()
    submission.to_csv('data/submit.csv', index=False)

train, labels, classes, test_ids, test = prepare_data(train, test)

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D

#experiment with dropouts
'''
img_model = Sequential()
img_model.add(Convolution2D(64, 5, 5, border_mode="same", input_shape=(1, img_height, img_width)))
img_model.add(Activation("relu"))
img_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2), border_mode="same"))
img_model.add(Convolution2D(32, 5, 5, border_mode="same"))
img_model.add(Activation("relu"))
img_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2), border_mode="same"))
img_model.add(Flatten())
img_model.add(Dense(1024))
img_model.add(Activation("relu"))
img_model.add(Dropout(0.5))

feat_model = Sequential()
feat_model.add(Dense(1024, input_dim=192))
feat_model.add(Activation("sigmoid"))
'''
model = Sequential()
#try dot product
#model.add(Merge([img_model, feat_model], mode='concat'))              
model.add(Dense(1024, input_dim=192))
model.add(Activation("sigmoid"))
model.add(Dense(512))
model.add(Activation("sigmoid"))
model.add(Dense(99))
model.add(Activation("softmax"))
if os.path.exists("weights.h5"):
    model.load_weights("weights.h5")
model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(train, labels, nb_epoch=100, batch_size=128)
model.save_weights("weights.h5")

preds = model.predict_proba(test)
make_submit(preds)



