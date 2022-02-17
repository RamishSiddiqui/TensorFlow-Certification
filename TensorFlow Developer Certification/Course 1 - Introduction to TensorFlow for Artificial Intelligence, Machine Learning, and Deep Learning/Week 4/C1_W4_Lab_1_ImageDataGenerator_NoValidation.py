from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, MaxPooling2D, Flatten, Conv2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing import image
import numpy as np
import easygui


def check_image():
    file_path = easygui.fileopenbox()

    img = image.load_img(file_path, target_size=(300, 300))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    print(classes[0])

    if classes[0] > 0.5:
        print('Human')
    else:
        print('Horse')


if __name__ == '__main__':
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    train_dir = 'data'
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(300, 300),
        batch_size=128,
        class_mode='binary'
    )

    # test_datagen = ImageDataGenerator(rescale=1. / 255)
    #
    # validation_generator = test_datagen.flow_from_directory(
    #     validation_dir,
    #     target_size=(300, 300),
    #     batch_size=32,
    #     class_mode='binary'
    # )

    model = Sequential([
        # Convolution Layers
        Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
        MaxPooling2D(2, 2),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        # MaxPooling2D(2, 2),
        # Conv2D(64, (3, 3), activation='relu'),
        # MaxPooling2D(2, 2),
        # Conv2D(64, (3, 3), activation='relu'),
        # MaxPooling2D(2, 2),

        # Dense Layers
        Flatten(),
        Dense(512, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer=RMSprop(learning_rate=0.001), loss='binary_crossentropy', metrics=['acc'])

    model.summary()

    history = model.fit(
        train_generator,
        steps_per_epoch=8,  # 1024 images / 128 batch == 8
        epochs=15,
        # validation_data=validation_generator,
        validation_steps=8,  # 256 images / 32 batch == 8
        verbose=1
    )

    check_image()
