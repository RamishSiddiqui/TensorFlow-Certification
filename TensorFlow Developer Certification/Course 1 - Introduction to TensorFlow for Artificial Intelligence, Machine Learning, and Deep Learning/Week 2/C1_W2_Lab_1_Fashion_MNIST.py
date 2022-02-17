from tensorflow.keras.datasets.fashion_mnist import load_data
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt


def display(data_set, index):
    plt.imshow(data_set[index], cmap='Greys')
    plt.show()


class myCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') >= 0.85:
            print('Accuracy Reached 85% Stopping the training process.')
            self.model.stop_training = True

if __name__ == '__main__':
    # Loading Data
    (X_train, y_train), (X_test, y_test) = load_data()

    # Normalizing data
    X_train, X_test = X_train / 255.0, X_test / 255.0

    # Displaying an image
    # display(X_train, 0)

    # Building Model
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation=tf.nn.relu),
        Dense(10, activation=tf.nn.softmax)
    ])

    # Compile model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Fitting the model
    model.fit(X_train, y_train, epochs=5, callbacks=[myCallback()])

    # Evaluate model
    eval = model.evaluate(X_test, y_test)
    print(eval)