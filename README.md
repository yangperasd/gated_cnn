#Gated CNN
This is Keras implementation of “Gated Linear Unit”.

#Requirements

 - Keras 2.1.2
 - Tensorflow 1.0.0
 - Others can be seen in requirements.txt

#Usage
The main Class is `GatedConvBlock` in `py/gated_cnn.py`.
Because there is a residual  connection in Gated Linear Unit (GLU), the padding of conv must be `same`. 
Let's take some example.
```
from gated_cnn import  GatedConvBlock
model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size,
                        padding='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))
model.add(GatedConvBlock(Convolution2D(nb_filters*2, kernel_size, 
                       padding='same')))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))
```
Check `py/mnist_gated_cnn.py` for more detail.

#Reference
- https://github.com/anantzoid/Language-Modeling-GatedCNN

> Written with [StackEdit](https://stackedit.io/).
