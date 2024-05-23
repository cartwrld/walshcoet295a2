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
print(crops_arr)

# create dict for the map to convert all vals to int at once
crop_to_int = {crop: i for i, crop in enumerate(crops_arr)}

df['Crop'] = df['Crop'].map(crop_to_int)

print(df.head().to_string())


# plt.figure(figsize=(10, 8))
# sb.heatmap(df.corr(), annot=True)
# plt.show()

print(df['Crop'].value_counts())


# def compile_model(optimizer, loss, title):
#     Y = df['Crop'].astype('float32')
#     X = df.drop("Crop", axis=1).astype('float32')
#
#     print(X.shape)
#
#     model = keras.Sequential()
#     # model.add(layers.Dense(256, activation="relu"))  # input layer
#     model.add(layers.Dense(32, activation="selu"))  # hidden layer 1
#     model.add(layers.Dense(16, activation="selu"))  # hidden layer 1
#     model.add(layers.Dense(7, activation="selu"))  # hidden layer 1
#     model.add(layers.Dense(4, activation="selu"))  # hidden layer 1
#     # model.add(layers.Dense(8, activation="selu"))  # hidden layer 2
#     model.add(layers.Dense(22, activation='softmax'))  # output layer
#     # model.add(layers.Dense(1, activation='sigmoid'))  # output layer
#
#     model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
#
#
#
#     results = model.fit(X, Y, validation_split=0.2, batch_size=32, epochs=50)
#     acc_chart(results, title)
#     loss_chart(results, title)

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

    results = model.fit(X, Y, validation_split=0.33, batch_size=10, epochs=50)

    acc_chart(results, title)
    loss_chart(results, title)

    test_predictions = [
        [90, 42, 43, 20.88, 82.00, 6.50, 202.93],
        [85, 58, 41, 21.77, 80.32, 7.04, 226.66],
        [60, 55, 44, 23.00, 82.32, 7.84, 263.96]
    ]

    print(test_predictions)


    x_pred = np.array([test_predictions[0]], dtype=np.float64)

    y_pred = (model.predict(x_pred) > 0.5).astype(int)

    print(y_pred[0])
    #
    # x_health = np.array([[50, 1, 2, 129, 196, 0, 0, 163, 0, 0, 0, 0, 0]], dtype=np.float64)
    # y_health = (model.predict(x_health) > 0.5).astype(int)
    # print(y_health[0])

    i = 0

    for sample in test_predictions:
        x_pred = np.array([sample], dtype=np.float64)

        y_pred = (model.predict(x_pred) > 0.5).astype(int)
        print(f"Sample {i}: {y_pred} => Top 2 recommended crops: {y_pred}")
        i += 1



compile_model("adam", "sparse_categorical_crossentropy", "Adam/BCE")

