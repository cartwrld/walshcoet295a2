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