import os
import tensorflow as tf
import pickle

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
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(lr=1e-4),
    metrics=["acc"],
)

# All images will be rescaled by 1./255
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255,
    # rotation_range=40,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    # shear_range=0.2,
    # zoom_range=0.2,
    # horizontal_flip=True,
    # fill_mode="nearest",
)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

# Flow training images in batches of 16 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    train_dir,  # This is the source directory for training images
    target_size=(150, 150),  # All images will be resized to 150x150
    batch_size=32,
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode="binary",
    classes=["cats", "dogs"],
)

# Flow validation images in batches of 16 using test_datagen generator
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary",
    classes=["cats", "dogs"],
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.1, patience=3, verbose=1
)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", min_delta=0, patience=10, verbose=1
)

history = model.fit_generator(
    train_generator,
    epochs=100,
    validation_data=validation_generator,
    verbose=1,
    callbacks=[reduce_lr, early_stopping],
)

model.save(os.path.join("models", "base_model.h5"))

with open(os.path.join("logs", "base_model_history.pck"), "wb") as file:
    pickle.dump(history.history, file)
