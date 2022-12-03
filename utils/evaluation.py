import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


def avg_accuracy(gt, pred):
    """
    gt: ground truth
    pred: prediction
    """
    correct_cnt = (pred == gt).sum()
    acc = correct_cnt * 1.0 / pred.shape[0]
    return acc


def class_metric(confusion_matrix, class_id):
    """
    confusion_matrix: confusion matrix of multi-class classification
    class_id: id of a particular class
    """
    confusion_matrix = np.float64(confusion_matrix)
    TP = confusion_matrix[class_id, class_id]
    FN = np.sum(confusion_matrix[class_id]) - TP
    FP = np.sum(confusion_matrix[:, class_id]) - TP
    TN = np.sum(confusion_matrix) - TP - FN - FP

    accuracy = (TP + TN) / (TP + FP + FN + TN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f_score = 2 * precision * recall / (precision + recall)
    return accuracy, precision, recall, f_score


# plot train loss
def visualize_train_loss(train_loss, logger, log_img_path):
    plt.xlabel('Train Loss')
    plt.plot(train_loss)
    path = log_img_path + 'train_loss.png'
    plt.savefig(path)
    plt.clf()
    logger.info('Train Loss Visualization is saved to ' + path)


# plot validation accuracy
def visualize_val_accuracy(val_acc, logger, log_img_path):
    plt.xlabel('Validation Accuracy')
    plt.plot(val_acc)
    path = log_img_path + 'val_acc.png'
    plt.savefig(path)
    plt.clf()
    logger.info('Validation Accuracy Visualization is saved to ' + path)


# plot confusion matrix
def visualize_confusion_matrix(gt, pred, logger, log_img_path):
    cm = confusion_matrix(gt, pred)
    disp = ConfusionMatrixDisplay(cm).plot()
    path = log_img_path + 'confusion_matrix.png'
    plt.savefig(path)
    plt.clf()
    logger.info('Confusion Matrix Visualization is saved to ' + path)
    return cm
