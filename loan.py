import keras
from keras import layers
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

from charts import acc_chart, loss_chart

df = pd.read_csv("data/loan.csv")

# (1) The Occupation Field is not needed for this (lots of Different occupations).
df.drop(['occupation'], axis=1, inplace=True)

# (2) The Categorized Strings should be converted to integer Vales for the analysis.
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})

for i, j in enumerate(df['education_level']):
    if j == 'High School':
        df.loc[i:i, 'education_level'] = 0
    elif j == "Bachelor's":
        df.loc[i:i, 'education_level'] = 1
    else:
        df.loc[i:i, 'education_level'] = 2

df['marital_status'] = df['marital_status'].map({'Married': 0, 'Single': 1})
df['loan_status'] = df['loan_status'].map({'Approved': 0, 'Denied': 1})

# # (3)[a] Create a Heatmap for the given DataFrame.
# plt.figure(figsize=(10, 8))
# sb.heatmap(df.corr(), annot=True)
# plt.show()
#
# # # (3)[b] Create Histograms that compare:
# loan_approved = df['loan_status'] == 0
# loan_denied = df['loan_status'] == 1
#
#
#
# # (3)[b]{i} Age against Approved/Denied
# plt.hist(df[loan_approved]['age'], color='b', alpha=0.5, bins=7, label="Approved", edgecolor='black')
# plt.hist(df[loan_denied]['age'], color='r', alpha=0.5, bins=3, label="Denied", edgecolor='black')
# plt.legend()
# plt.ylabel("Number of Applicants")
# plt.xlabel("Applicant Age")
# plt.title("Loan Status -vs- Age")
# plt.show()
#
# # (3)[b]{ii} Education against Approved/Denied
# bins = np.arange(0, 4) - 0.5
# plt.hist(df[loan_approved]['education_level'], color='b', alpha=0.5, bins=bins, label="Approved", edgecolor='black')
# plt.hist(df[loan_denied]['education_level'], color='r', alpha=0.5, bins=bins, label="Denied", edgecolor='black')
# plt.legend()
# plt.xticks(range(0, 3))
# # plt.yticks(np.arange(0, 30))
# plt.ylabel("Number of Applicants")
# plt.xlabel("Education Level")
# plt.title("Loan Status -vs- Education Level")
# plt.show()
#
#
# # (3)[b]{iii} Married/Single against Approved Denied
# bins = np.arange(0, 3) - 0.5
# plt.hist(df[loan_approved]['marital_status'], color='b', alpha=0.5, bins=bins, label="Approved", edgecolor='black')
# plt.hist(df[loan_denied]['marital_status'], color='r', alpha=0.5, bins=bins, label="Denied", edgecolor='black')
# plt.legend()
# plt.ylabel("Number of Applicants")
# plt.xlabel("Marital Status")
# plt.title("Loan Status -vs- Marital Status")
# plt.xticks(range(0, 2))
# plt.show()

#  (4) Create an appropriate mode given the Modified DataFrame you have prepared.
Y = df['loan_status'].astype('float32')
X = df.drop("loan_status", axis=1).astype('float32')

model = keras.Sequential()
model.add(layers.Dense(7, activation="relu"))  # input layer
# model.add(layers.Dense(5, activation="relu"))  # input layer
model.add(layers.Dense(3, activation="relu"))  # input layer
model.add(layers.Dense(1, activation="sigmoid"))  # hidden layer 1
# model.add(layers.Dense(4, activation="relu"))  # hidden layer 2
# model.add(layers.Dense(1))  # output layer

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

results = model.fit(X, Y, validation_split=0.21, batch_size=100, epochs=200)

acc_chart(results)
loss_chart(results)

model.save("models/loan.keras")











