import numpy as np
import os
import tensorflow as tf
from glob import glob
from PIL import Image
import cv2
import pickle
from encoder import encoder

k = 4
img_size = 150

base_dir = "cats_and_dogs_filtered"
train_dir = os.path.join(base_dir, "train")
validation_dir = os.path.join(base_dir, "validation")

# Directory with our training cat pictures
train_cats_dir = os.path.join(train_dir, "cats")

# Directory with our training dog pictures
train_dogs_dir = os.path.join(train_dir, "dogs")

# Directory with our validation cat pictures
validation_cats_dir = os.path.join(validation_dir, "cats")

# Directory with our validation dog pictures
validation_dogs_dir = os.path.join(validation_dir, "dogs")

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(150, 150, 3)
        ),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(k + 1, activation="softmax"),
    ]
)

model.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer=tf.keras.optimizers.Adam(lr=1e-4),
    metrics=["acc"],
)

### data generator ###
def data_generator(cats_dir: str, dogs_dir: str, batch_size):
    cats_list = glob(os.path.join(cats_dir, "*.jpg"))
    dogs_list = glob(os.path.join(dogs_dir, "*.jpg"))
    while True:
        x_data = []
        y_data = []
        for _ in range(batch_size):
            cat_num = np.random.randint(0, k + 1)
            dog_num = k - cat_num
            data_list = np.hstack(
                [
                    np.random.choice(cats_list, cat_num),
                    np.random.choice(dogs_list, dog_num),
                ]
            )
            np.random.shuffle(data_list)
            x = encoder.encoder(data_list, 0)
            y = np.eye(k + 1, k + 1)[dog_num]
            x_data.append(x)
            y_data.append(y)
        yield np.array(x_data), np.array(y_data)


# class SequenceData(tf.keras.utils.Sequence):
#     def __init__(
#         self,
#         datagen: tf.keras.preprocessing.image.ImageDataGenerator,
#         dataset_dir: str,
#         batch_size: int,
#     ):
#         self.generator = datagen.flow_from_directory(
#             dataset_dir,  # This is the source directory for training images
#             batch_size=k,
#             # Since we use binary_crossentropy loss, we need binary labels
#             class_mode="binary",
#             classes=["cats", "dogs"],
#         )
#         self.batch_size = batch_size

#     # 返回长度，通过len(<你的实例>)调用
#     def __len__(self):
#         return len(self.generator) // self.batch_size

#     # 即通过索引获取a[0],a[1]这种
#     def __getitem__(self, idx):
#         if idx == 0:
#             self.on_epoch_end()
#         x_data = []
#         y_data = []
#         for b in range(self.batch_size):
#             x, y = self.generator[idx * self.batch_size + b]
#             x = get_X(x)
#             y = np.eye(k + 1, k + 1)[int(np.sum(y))]
#             x_data.append(x)
#             y_data.append(y)
#         print(y_data)
#         return np.array(x_data), np.array(y_data)

#     def on_epoch_end(self):
#         self.generator.on_epoch_end()


# train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
#     # rotation_range=40,
#     # width_shift_range=0.2,
#     # height_shift_range=0.2,
#     # shear_range=0.2,
#     # zoom_range=0.2,
#     # horizontal_flip=True,
#     # fill_mode="nearest",
# )
# test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
#     # rotation_range=40,
#     # width_shift_range=0.2,
#     # height_shift_range=0.2,
#     # shear_range=0.2,
#     # zoom_range=0.2,
#     # horizontal_flip=True,
#     # fill_mode="nearest",
# )


train_generator = data_generator(train_cats_dir, train_dogs_dir, 32)
validation_generator = data_generator(validation_cats_dir, validation_dogs_dir, 32)
# next(train_generator)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.1, patience=3, verbose=1
)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", min_delta=0, patience=10, verbose=1
)

history = model.fit(
    train_generator,
    steps_per_epoch=500,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=50,
    verbose=1,
    callbacks=[reduce_lr, early_stopping],
)

model.save(os.path.join("models", f"parity_model_ks{k}.h5"))

with open(os.path.join("logs", f"parity_model_history_ks{k}.pck"), "wb") as file:
    pickle.dump(history.history, file)
