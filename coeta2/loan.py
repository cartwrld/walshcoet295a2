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
# print(df[0])

# df_copy = df
#
# df['gen'] = np.NAN

for i, j in enumerate(df['gender']):
    if j == 'Male':
        df.loc[i:i, 'gender'] = 0
    else:
        df.loc[i:i, 'gender'] = 1

for i, j in enumerate(df['education_level']):
    if j == 'High School':
        df.loc[i:i, 'education_level'] = 0
    elif j == "Bachelor's":
        df.loc[i:i, 'education_level'] = 1
    else:
        df.loc[i:i, 'education_level'] = 2

for i, j in enumerate(df['marital_status']):
    if j == 'Single':
        df.loc[i:i, 'marital_status'] = 0
    elif j == 'Married':
        df.loc[i:i, 'marital_status'] = 1

for i, j in enumerate(df['loan_status']):
    if j == 'Denied':
        df.loc[i:i, 'loan_status'] = 0
    elif j == 'Approved':
        df.loc[i:i, 'loan_status'] = 1

sb.heatmap(df.corr(), annot=True)
plt.show()

df['gender'] = df['gender'].map({0: "Male", 1: "Female"})
df['loan_status'] = df['loan_status'].map({0: "Approved", 1: "Denied"})

loan_approved = df['loan_status'] == 'Approved'
loan_denied = df['loan_status'] == 'Denied'

# compare age with health
plt.hist(df[loan_approved]['age'], color='b', alpha=0.5, bins=15, label="Approved")
plt.hist(df[loan_denied]['age'], color='r', alpha=0.5, bins=15, label="Denied")
plt.legend()
plt.title("Health Count -vs- Age")
plt.show()

print(df.head().to_string())
