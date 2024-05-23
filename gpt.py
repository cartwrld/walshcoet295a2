import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("data/Crop_Recommendation.csv")

# Encode the labels as integers
crops_arr = np.unique(df['Crop'])
crop_to_int = {crop: i for i, crop in enumerate(crops_arr)}
df['Crop'] = df['Crop'].map(crop_to_int)

# Check the label distribution
print(df['Crop'].value_counts())

# Separate features and labels
Y = df['Crop'].astype('int32')
X = df.drop("Crop", axis=1).astype('float32')

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)


# Define model architecture
def create_model():
    model = keras.Sequential([
        layers.Dense(64, activation="relu", input_shape=(X.shape[1],), kernel_regularizer=l2(0.01)),
        layers.Dropout(0.5),
        layers.Dense(32, activation="relu", kernel_regularizer=l2(0.01)),
        layers.Dropout(0.5),
        layers.Dense(len(crops_arr), activation="softmax")
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# Function to plot accuracy
def acc_chart(history, title):
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.title(f'{title} Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


# Function to plot loss
def loss_chart(history, title):
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.title(f'{title} Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


# K-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

val_accuracies = []
val_losses = []

for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index], X[val_index]
    Y_train, Y_val = Y[train_index], Y[val_index]

    model = create_model()

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=50, batch_size=32,
                        callbacks=[early_stopping])

    val_accuracies.append(history.history['val_accuracy'])
    val_losses.append(history.history['val_loss'])

    acc_chart(history, "Adam/SCCE")
    loss_chart(history, "Adam/SCCE")

# Average validation accuracy and loss over all folds
avg_val_accuracy = np.mean([acc[-1] for acc in val_accuracies])
avg_val_loss = np.mean([loss[-1] for loss in val_losses])

print(f"Average Validation Accuracy: {avg_val_accuracy}")
print(f"Average Validation Loss: {avg_val_loss}")
