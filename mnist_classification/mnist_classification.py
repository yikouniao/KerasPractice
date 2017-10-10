# mnist classification by cnn

import keras
from keras.layers import Input, Conv2D, MaxPool2D, Dropout, Flatten, Dense
from keras.models import Model

# 60,000 28*28 greyscale images of 10 digits, along with a test set of 10,000 images
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

img_rows, img_cols, class_num = 28, 28, 10
batch_size, epochs = 128, 12

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, class_num)
y_test = keras.utils.to_categorical(y_test, class_num)

# the nn model
# not channels-first
inputs = Input(shape=(img_rows, img_cols, 1))
x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid',
           data_format='channels_last', activation='relu')(inputs)
x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid',
           data_format='channels_last', activation='relu')(x)
x = MaxPool2D(pool_size=(2, 2), strides=None, padding='valid',
           data_format='channels_last')(x)
x = Dropout(0.25)(x)
x = Flatten()(x)
x = Dense(units=128, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(units=class_num, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=keras.optimizers.Adadelta(),
              loss=keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])