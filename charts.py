import matplotlib.pyplot as plt

def acc_chart(results, title=""):
    # plt.title("Accuracy of Model")
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.plot(results.history['accuracy'])
    plt.plot(results.history['val_accuracy'])
    plt.legend(['train', 'test'], loc="upper left")
    plt.title(title + " Accuracy")
    plt.show()


def loss_chart(results, title=""):
    # plt.title("Model Losses")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.plot(results.history['loss'])
    plt.plot(results.history['val_loss'])
    plt.legend(['train', 'test'], loc="upper left")
    plt.title(title + " Losses")
    plt.show()