import os
import pandas as pd
import numpy as np
import pandas as pd 
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import json
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization


IMAGE_WIDTH= 64
IMAGE_HEIGHT= 64
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3

def predict():
    filenames = os.listdir("../finall/char/")
    categories = []
    pre = [0*i for i  in range(len(filenames))]

    for filename in filenames:
        categories.append(filename.split('.')[0])


    df = pd.DataFrame({
        'filename': filenames,
        'category': categories,
        'pre': pre
    })

    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(13, activation='softmax')) # 2


    data = ImageDataGenerator(rescale= 1./ 255)
    data_generator = data.flow_from_dataframe(
        df,
        "../finall/char/",
        x_col='filename',
        y_col=None,
        class_mode=None,
        target_size=IMAGE_SIZE,
        shuffle=False
    )

    model.load_weights('model.h5')

    predict = model.predict_generator(data_generator)
    df['pre'] = np.argmax(predict, axis=-1)

    df['pre'] = df['pre'].replace({10: "A", 11 : "E", 12 : "F"})
    df = df.sort_values(by="category")
    result = []
    list_ = df['pre'].values
    for i in range(len(list_)):
        result.append(str(list_[i]))
    return ' '.join(result)





