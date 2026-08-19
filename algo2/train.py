"""
Basic CNN to classify cracked vs not-cracked concrete images.

Expected folder structure:
    data/
        Positive/   -> cracked images
        Negative/   -> non-cracked images
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------- Settings ----------------
DATA_DIR = "test_data"        # path to your folder
IMG_SIZE = (128, 128)    # resize images (smaller = faster)
BATCH_SIZE = 32
EPOCHS = 5
SEED = 123

# ---------------- Load data ----------------
# 80% train, 20% validation
train_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
)

val_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
)

class_names = train_ds.class_names
print("Classes:", class_names)   # e.g. ['Negative', 'Positive'] -> 0, 1

# Speed up loading
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

# ---------------- Build model ----------------
model = keras.Sequential([
    layers.Input(shape=IMG_SIZE + (3,)),
    layers.Rescaling(1.0 / 255),          # scale pixels to 0-1

    layers.Conv2D(16, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid"),  # 1 output: probability of "cracked"
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ---------------- Train ----------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
)

# ---------------- Save ----------------
model.save("crack_model.keras")
print("Model saved to crack_model.keras")
