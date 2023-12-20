import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate, BatchNormalization, Dropout
from tensorflow.keras.models import Model
import numpy as np

class NN:
    def __init__(self, input_shape, dim_of_policy, dim_of_value):
        self.input_shape = input_shape
        self.dim_of_policy = dim_of_policy
        self.dim_of_value = dim_of_value
        self.model = self.build_model()


    def build_model(self):
        input_layer = Input(shape=self.input_shape)

        # Shared convolutional layers
        num_filters = 32
        filter_size = 1
        pool_size = 2
        conv1 = Conv2D(num_filters, filter_size, activation='relu')(input_layer)
        batch = BatchNormalization()(conv1)
        drop1 = Dropout(0.2)(batch)
        maxpool1 = MaxPooling2D(pool_size=pool_size)(drop1)
        conv2 = Conv2D(64, 1, activation='relu')(maxpool1)
        drop2 = Dropout(0.1)(conv2)
        maxpool2 = MaxPooling2D(1)(drop2)
        flatten = Flatten()(maxpool2)

        # Head 1 for image classification
        dense1_head1 = Dense(128, activation='relu')(flatten)
        output_head1 = Dense(self.dim_of_policy, activation='softmax', name="output_head1")(dense1_head1)

        # Head 2 for object detection
        dense1_head2 = Dense(128, activation='relu')(flatten)
        output_head2 = Dense(self.dim_of_value, activation='tanh', name="output_head2")(dense1_head2) #between -1 and 1
        # output_head2 = Dense(1, activation='linear')(dense1_head2)

        # Create a merged model
        # merged_model = concatenate([output_head1, output_head2], axis=1)   
        model = Model(inputs=input_layer, outputs=[output_head1, output_head2])

        return model

    def compile_model(self, loss_classification, loss_detection, optimizer):
        self.model.compile(
            loss={'output_head1': loss_classification, 'output_head2': loss_detection},
            optimizer=optimizer,
            metrics={'output_head1': 'accuracy', 'output_head2': 'mean_squared_error'} #precision
        )

    def fit_model(self, x_train, y_train_head1, y_train_head2, epochs, batch_size):
        history= self.model.fit(x_train, {'output_head1': y_train_head1, 'output_head2': y_train_head2},
                       epochs=epochs, batch_size=batch_size)
        return history
        print(history.history['loss'])

# Usage example:
# input_shape = (4, 4, 2)
# dim_of_policy = 21
# dim_of_value = 1

# two_headed_cnn = NN(input_shape, dim_of_policy, dim_of_value)
# two_headed_cnn.compile_model(loss_classification='categorical_crossentropy',
#                        loss_detection='mean_squared_error',
#                        optimizer='adam')

# data = np.ones((15,4,4,2))
# y1 = np.ones((15, 21))
# y2 = np.ones((15, 1))


# two_headed_cnn.fit_model(data, y1, y2, epochs=10, batch_size=32)
# data1 = np.ones((2,4,4,2))
# a = two_headed_cnn.model.predict(data1)
# print(a[0])
# print(a[1])
