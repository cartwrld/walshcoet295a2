import keras
from keras import layers, models
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from charts import acc_chart, loss_chart
from tensorflow.keras.regularizers import l2

df = pd.read_csv("data/Crop_Recommendation.csv")

crops_arr = np.unique(df['Crop'])

# create dict for the map to convert all vals to int at once
crop_dict = {crop: i for i, crop in enumerate(crops_arr)}

df['Crop'] = df['Crop'].map(crop_dict)

# plt.figure(figsize=(10, 8))
# sb.heatmap(df.corr(), annot=True)
# plt.show()


def compile_model(optimizer, loss, title):
    Y = df['Crop'].astype('float32')
    X = df.drop("Crop", axis=1).astype('float32')

    model = keras.Sequential([
        layers.Dense(7, activation="relu", kernel_regularizer=l2(0.01)),
        # layers.Dropout(0.5),
        layers.Dense(11, activation="leaky_relu", kernel_regularizer=l2(0.01)),
        # layers.Dense(4, activation="leaky_relu", kernel_regularizer=l2(0.01)),
        # layers.Dropout(0.5),
        layers.Dense(len(crops_arr), activation="softmax"),
    ])

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    results = model.fit(X, Y, validation_split=0.33, batch_size=10, epochs=10)

    acc_chart(results, title)
    loss_chart(results, title)

    test_predictions = np.array([
        np.array([90, 42, 43, 20.88, 82.00, 6.50, 202.93]),
        np.array([85, 58, 41, 21.77, 80.32, 7.04, 226.66]),
        np.array([60, 55, 44, 23.00, 82.32, 7.84, 263.96])
    ])

    get_top_crops(test_predictions)
    # print(test_predictions)
    #
    # x_pred = np.array([test_predictions[0]], dtype=np.float64)
    #
    # y_pred = (model.predict(x_pred) > 0.5).astype(int)
    #
    # print(y_pred[0])



def get_top_crops(preds):
    """This function will take in a numpy array and determine the 2 highest
    values and print out the corresponding crop strings"""
    print(f"probabilities = {preds}")

    first_place = preds.max()
    print(first_place)
    # index = crop_dict[]
    highest_index = np.where(preds == first_place)[0][0]
    highest_string = crop_dict[highest_index]

    preds[highest_index] = -1

    second_highest = max(preds)
    second_highest_index = np.where(preds == second_highest)[0][0]
    second_highest_string = crop_dict[second_highest_index]

    print(f"highest prob = {first_place}")
    print(f"highest_index = {highest_index}")
    print(f"highest_string = {highest_string}")

    print(f"2nd highest prob = {second_highest}")
    print(f"2nd highest_index = {second_highest_index}")
    print(f"2nd highest_string = {second_highest_string}")


compile_model("adam", "sparse_categorical_crossentropy", "Adam/BCE")
