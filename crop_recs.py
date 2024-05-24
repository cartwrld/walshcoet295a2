import keras
from keras import layers, models
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from charts import acc_chart, loss_chart


df = pd.read_csv("data/Crop_Recommendation.csv")

crops_arr = np.unique(df['Crop'])

# create dict for the map to convert all vals to int at once
crop_dict = {crop: i for i, crop in enumerate(crops_arr)}

print(crop_dict)

df['Crop'] = df['Crop'].map(crop_dict)

plt.figure(figsize=(20, 12))
sb.heatmap(df.corr(), annot=True)
plt.show()


def compile_model(optimizer, loss, title):
    X = df.drop("Crop", axis=1).astype('float32')
    Y = df['Crop'].astype('float32')

    model = keras.Sequential([

        layers.Dense(512, activation="leaky_relu"),
        layers.Dropout(0.6),
        layers.Dense(256, activation="leaky_relu"),
        layers.Dropout(0.2),



        layers.Dense(len(crops_arr), activation="softmax"),

    ])

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    results = model.fit(X, Y, validation_split=0.211, batch_size=11, epochs=50)

    acc_chart(results, title)
    # loss_chart(results, title)

    test_predictions = [
        np.array([90, 42, 43, 20.88, 82.00, 6.50, 202.93]),
        np.array([85, 58, 41, 21.77, 80.32, 7.04, 226.66]),
        np.array([60, 55, 44, 23.00, 82.32, 7.84, 263.96])
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
                # print(f"first_index: {first_index}")
                # print(f"first_highest: {first_highest}")
                # print(f"kv: {key}:{value}")
                return [first_index, first_highest, key]

        return None

    details = get_highest_details()

    first_crop = details[2]
    # print(first_crop)
    print("=============================================================")
    print(f"1st Place Crop: {first_crop}, Probability: {details[1]:.4f}")

    preds[details[0]] = -1

    details = get_highest_details()
    second_crop = details[2]
    print(f"2nd Place Crop: {second_crop}, Probability: {details[1]:.4f}")


compile_model("adam", "sparse_categorical_crossentropy", "Adam/SCCE")
