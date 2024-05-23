import keras
from keras import layers, models
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

from charts import acc_chart, loss_chart

df = pd.read_csv("data/Crop_Recommendation.csv")

crops_arr = np.unique(df['Crop'])
print(crops_arr)

# create dict for the map to convert all vals to int at once
crop_to_int = {crop: i for i, crop in enumerate(crops_arr)}

df['Crop'] = df['Crop'].map(crop_to_int)

# plt.figure(figsize=(10, 8))
# sb.heatmap(df.corr(), annot=True)
# plt.show()


Y = df['Crop'].astype('float32')
X = df.drop("Crop", axis=1).astype('float32')

print(X.shape)

model = keras.Sequential()
model.add(layers.Dense(7, activation="relu"))  # input layer
model.add(layers.Dense(14, activation="relu"))  # input layer
# model.add(layers.Dense(3, activation="relu"))  # input layer
model.add(layers.Dense(1, activation="relu"))  # hidden layer 1
# model.add(layers.Dense(4, activation="relu"))  # hidden layer 2
# model.add(layers.Dense(1))  # output layer

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

results = model.fit(X, Y, validation_split=0.33, batch_size=32, epochs=10)

acc_chart(results)
loss_chart(results)

X_at_risk = np.array([[62, 1, 3, 145, 250, 1, 2, 120, 0, 1.4, 1, 1, 0]], dtype=np.float64)
y_at_risk = (model.predict(X_at_risk) > 0.5).astype(int)

print(y_at_risk[0])

x_health = np.array([[50, 1, 2, 129, 196, 0, 0, 163, 0, 0, 0, 0, 0]], dtype=np.float64)
y_health = (model.predict(x_health) > 0.5).astype(int)
print(y_health[0])
