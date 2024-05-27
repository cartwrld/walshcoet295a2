import keras
from keras import layers
import pandas as pd
import numpy as np
import tensorflow as tf
from charts import acc_chart, loss_chart

# # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                   #
#   Name:            Carter Walsh (walsh0715)       #
#   Class:           COET295 - Assignment 2         #
#   Instructor:      Bryce Barrie & Wade Lahoda     #
#   Date:            Monday, May 27th, 2024         #
#                                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # #

df = pd.read_csv("data/Crop_Recommendation.csv")

crops_arr = np.unique(df['Crop'])

# create dict for the map to convert vals to int all at once
crop_dict = {crop: i for i, crop in enumerate(crops_arr)}

df['Crop'] = df['Crop'].map(crop_dict)


def compile_model(optimizer, loss, title, vs):
    X = df.drop("Crop", axis=1).astype('float32')
    Y = df['Crop'].astype('float32')

    model = keras.Sequential([
        layers.Dense(32, activation="leaky_relu"),
        layers.Dropout(0.2),
        layers.Dense(len(crops_arr), activation="softmax"),
    ])

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    results = model.fit(X, Y, validation_split=vs, batch_size=13, epochs=200, shuffle=True)

    acc_chart(results, title)
    loss_chart(results, title)

    test_predictions = [
        np.array([108, 22, 46, 26.3745356, 86.3745356, 6.357345, 53.45757]),          # watermelon
        np.array([70, 68, 45, 33.745754365, 92.34576436, 6.4574587, 203.345753754]),  # papaya
        np.array([35, 142, 203, 21.345756345, 90.3456347, 5.456356, 123.34563575])    # apple
    ]

    for test_pred in test_predictions:
        test_pred = test_pred.reshape(1, -1)
        test_model_preds = model.predict(test_pred).astype("float32")
        get_top_crops(test_model_preds[0], crop_dict)


def get_top_crops(preds, crops):
    """This function takes in a numpy array of probabilities and prints out the names
    and probabilities of the top two crops."""

    preds = tf.nn.softmax(preds).numpy()  # Convert logits to probabilities

    def get_highest_details():
        """This function takes in a numpy array of probabilities and prints out
        the names of the top two crops."""

        first_index = np.argmax(preds)
        first_highest = preds[first_index]

        for key, value in crops.items():
            if value == first_index:
                return [first_index, first_highest * 1000, key]

        return None

    details = get_highest_details()
    first_crop = details[2]
    print("=============================================")
    print(f"1st Place Crop: {first_crop}, Probability: {details[1]:.4f}%")

    preds[details[0]] = -1

    details = get_highest_details()
    second_crop = details[2]
    print(f"2nd Place Crop: {second_crop}, Probability: {details[1]:.4f}%")


compile_model("adam", "sparse_categorical_crossentropy", "Adam/SCCE", 0.041)
