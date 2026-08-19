import numpy as np
import tensorflow as tf
from tensorflow import keras

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
CLASS_NAMES = ["Negative", "Positive"] 

model = keras.models.load_model("crack_model.keras")

TEST_DIR = "test_data"

test_ds = keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False,
)

loss, acc = model.evaluate(test_ds)
print(f"Test loss: {loss:.4f}  |  Test accuracy: {acc:.4f}")


y_true = np.concatenate([y.numpy() for _, y in test_ds]).ravel()
y_prob = model.predict(test_ds).ravel()
y_pred = (y_prob > 0.5).astype(int)

cm = tf.math.confusion_matrix(y_true, y_pred).numpy()
print("Confusion matrix [rows=true, cols=pred]:")
print(cm)


def predict_image(path):
    img = keras.utils.load_img(path, target_size=IMG_SIZE)
    arr = keras.utils.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)  

    prob = float(model.predict(arr, verbose=0)[0][0])
    label = CLASS_NAMES[int(prob > 0.5)]
    print(f"{path} -> {label}  (probability of crack: {prob:.3f})")
    return label, prob


if __name__ == "__main__":
    predict_image("sample.jpg")