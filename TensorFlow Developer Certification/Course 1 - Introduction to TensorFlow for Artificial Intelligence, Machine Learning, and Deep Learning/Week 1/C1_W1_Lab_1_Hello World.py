from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

if __name__ == '__main__':
    X = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    Y = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

    model = Sequential([Dense(units=1, input_shape=[1])])
    model.compile(optimizer='sgd', loss='mean_squared_error')

    model.fit(X, Y, epochs=500)

    pred = model.predict([10.0])

    print(pred)