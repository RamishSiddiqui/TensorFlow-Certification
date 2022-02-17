import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.datasets.mnist import load_data


class myCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') >= 0.99:
            print('\nAccuracy Reached 99% Stopping the training process.')
            self.model.stop_training = True


if __name__ == '__main__':
    # Loading Data
    (X_train, y_train), (X_test, y_test) = load_data()

    # Normalizing data
    X_train, X_test = X_train / 255.0, X_test / 255.0

    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    # Displaying an image
    # display(X_train, 0)

    # Make model
    model = Sequential([
        # Convolution Layers
        Conv2D(64, (3, 3), activation=tf.nn.relu, input_shape=(28, 28, 1)),
        MaxPooling2D(2, 2),
        # Conv2D(64, (3, 3), activation=tf.nn.relu),
        # MaxPooling2D(2, 2),
        # Conv2D(64, (3, 3), activation=tf.nn.relu),
        # MaxPooling2D(2, 2),

        # Dense Layers
        Flatten(),
        Dense(128, activation=tf.nn.relu),
        Dense(10, activation=tf.nn.softmax)
    ])

    model.summary()

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=10, callbacks=[myCallback()])

    loss = model.evaluate(X_test, y_test)

    print(loss)
