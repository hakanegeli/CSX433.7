import matplotlib.pyplot as plt
import numpy as np
import itertools

"""
Some helper function to plot data 
"""


def plot_data(x, y, epochs):
    """
    This function plots the model loss over the iterations.
    """

    fig = plt.figure()
    ax = fig.gca()

    ax.set_ylim(0, int(np.max(y)+0.5))
    ax.set_xlim(0, np.max(x))
    ax.yaxis.grid(True)
    ax.grid(which='minor', axis='x', alpha=0.2)
    ax.grid(which='major', axis='x', alpha=0.5)
    major_ticks = np.arange(0, np.max(x), 88)
    minor_ticks = np.arange(0, np.max(x), 16)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)

    fig.canvas.draw()
    labels = ["{:2d}".format(int(int(item.get_text())/88)) for item in ax.get_xticklabels()]
    ax.set_xticklabels(labels)

    plt.title("Model Loss over {} Epochs".format(epochs))
    plt.scatter(x, y, s=50, alpha=0.5, label='cross_entropy')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='upper right')
    plt.show()


def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """

    plt.figure()

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
