import keras
from keras import layers
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt


def acc_chart(results):
    plt.title("Accuracy of Model")
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.plot(results.history['accuracy'])
    plt.plot(results.history['val_accuracy'])
    plt.legend(['train', 'test'], loc="upper left")
    plt.show()


def loss_chart(results):
    plt.title("Model Losses")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.plot(results.history['loss'])
    plt.plot(results.history['val_loss'])
    plt.legend(['train', 'test'], loc="upper left")
    plt.show()


df = pd.read_csv("data/loan.csv")

df.drop(['occupation'], axis=1, inplace=True)



print(df.head().to_string())

print("\n\n Shape and Size")
print(df.shape)

print(df.dtypes)

print(type(df))
print(df[0])

df_gender = df[df['gender'] == 'Male']
