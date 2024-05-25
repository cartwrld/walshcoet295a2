import keras
from keras import layers, models
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from charts import acc_chart, loss_chart
import random as randy

df = pd.read_csv("data/Crop_Recommendation.csv")

crops_arr = np.unique(df['Crop'])

# create dict for the map to convert vals to int all at once
crop_dict = {crop: i for i, crop in enumerate(crops_arr)}


# print(df.head(200).to_string())
df['Crop'] = df['Crop'].map(crop_dict)
# print(df.head(200).to_string())

# plt.figure(figsize=(20, 12))
# sb.heatmap(df.corr(), annot=True)
# plt.show()


def compile_model(optimizer, loss, title, vs):
    data_augmentation = keras.Sequential([

    ])

    X = df.drop("Crop", axis=1).astype('float32')
    Y = df['Crop'].astype('float32')

    model = keras.Sequential([
        layers.Dense(32, activation="leaky_relu"),
        layers.Dropout(0.2),
        # layers.Dense(16, activation="leaky_relu"),
        # layers.Dropout(0.1),
        # layers.Dense(15, activation="leaky_relu"),
        layers.Dense(len(crops_arr), activation="softmax"),

    ])

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # model.save("models/crop.keras")

    results = model.fit(X, Y, validation_split=vs, batch_size=13, epochs=200, shuffle=True)

    acc_chart(results, title)
    loss_chart(results, title)

    test_predictions = [
        np.array([90., 42., 43., 20.88, 82.00, 6.50, 202.93]),
        np.array([85., 58., 41., 21.77, 80.32, 7.04, 226.66]),
        np.array([60., 55., 44., 23.00, 82.32, 7.84, 263.96]),
        np.array([87., 41., 45., 24.55, 80.26, 7.06, 246.87]),
        np.array([82., 57., 44., 23.98, 80.01, 6.15, 219.93]),
        np.array([79., 56., 48., 21.11, 84.75, 7.44, 246.35]),
        np.array([83., 54., 46., 23.17, 81.91, 6.25, 253.13]),
        np.array([55., 50., 49., 23.82, 82.84, 7.35, 228.75]),
        np.array([99., 42., 42., 21.7, 82.23, 7.88, 233.9]),
        np.array([77., 51., 46., 23.76, 83.41, 6.34, 260.32])

    ]

    for test_pred in test_predictions:
        get_top_crops(test_pred,crop_dict)


def get_top_crops(preds, crop_dict):
    """This function takes in a numpy array of probabilities and prints out the names
    and probabilities of the top two crops."""

    def get_highest_details():
        """This function takes in a numpy array of probabilities and prints out
        the names of the top two crops."""

        first_index = np.argmax(preds)
        first_highest = preds[first_index]

        for key, value in crop_dict.items():
            if value == first_index:
                return [first_index, first_highest, key]

        return None

    details = get_highest_details()

    first_crop = details[2]
    # print(first_crop)
    print("=============================================")
    print(f"1st Place Crop: {first_crop}, Probability: {details[1]:.4f}")

    preds[details[0]] = -1

    details = get_highest_details()
    second_crop = details[2]
    print(f"2nd Place Crop: {second_crop}, Probability: {details[1]:.4f}")


# compile_model("adam", "sparse_categorical_crossentropy", "Adam/SCCE", 0.01)
# compile_model("adam", "sparse_categorical_crossentropy", "Adam/SCCE", 0.02)
# compile_model("adam", "sparse_categorical_crossentropy", "Adam/SCCE", 0.03)
compile_model("adam", "sparse_categorical_crossentropy", "Adam/SCCE", 0.041)
# compile_model("adam", "sparse_categorical_crossentropy", "Adam/SCCE", 0.05)
# compile_model("adam", "sparse_categorical_crossentropy", "Adam/SCCE", 0.06)
# compile_model("adam", "sparse_categorical_crossentropy", "Adam/SCCE", 0.07)
# compile_model("adam", "sparse_categorical_crossentropy", "Adam/SCCE",0.08 )
# compile_model("adam", "sparse_categorical_crossentropy", "Adam/SCCE", 0.09)
#
# for i in range(10,100):
#     compile_model("adam", "sparse_categorical_crossentropy", "Adam/SCCE", i/100)







