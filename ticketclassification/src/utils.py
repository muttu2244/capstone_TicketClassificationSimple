import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(labels, predictions, label_names):
    """
    Plots a confusion matrix.
    Args:
        labels (list): True labels.
        predictions (list): Predicted labels.
        label_names (list): Names of the labels.
    """
    cm = confusion_matrix(labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()
