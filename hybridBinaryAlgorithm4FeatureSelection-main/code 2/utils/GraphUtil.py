"""
Created on Fri Mar  5 10:42:58 2021

@author: Oyelade
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import DetCurveDisplay, RocCurveDisplay, ConfusionMatrixDisplay
#from sklearn.metrics.ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix#, plot_confusion_matrix
import pandas as pd

def _draw_rates__(data=None, filename=None, pathsave=None):
    for d in data:
        x = np.arange(len(d[0]))
        print(d[1]+'   '+str(d[0]))
        plt.plot(x, d[0], label=d[1])       # e.g infection rate
    
    plt.ylabel('Population')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.savefig(pathsave + filename + ".png")
    plt.close()
    return None

def draw_raw_time_series_data_and_show(data=None, label=None, title=None):
    plt.plot(data)
    plt.xlabel(label["y"])
    plt.ylabel(label["x"])
    plt.title(title, fontsize=8)
    plt.show()
    return None

def _plot_training_result__(loss, val_loss, accuracy, val_accuracy, epochs, experiment, pathsave):
    N = epochs
    plt.style.use('seaborn-whitegrid')
    plt.title("Training/Validation Loss")
    plt.plot(np.arange(0, N),loss, label='training')
    plt.plot(np.arange(0, N),val_loss, label='validation')
    plt.ylabel("Loss")
    plt.xlabel("Epoch #")
    plt.legend(['training', 'validation'], loc="lower left")
    plt.savefig(pathsave +experiment+ "_loss.png")
    plt.show()
        
    plt.title("Training/Validation Accuracy")
    plt.plot(np.arange(0, N),accuracy, label="train_accuracy")
    plt.plot(np.arange(0, N),val_accuracy, label="validation_accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch #")
    plt.legend(['training', 'validation'], loc="lower left")
    plt.savefig(pathsave+experiment + "_acc.png")
    plt.show()

def _draw_score_confusion_matrix__(data=None, filename=None, pathsave=None):
    plt.imshow(data, cmap=plt.cm.Blues)
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.xticks([], [])
    plt.yticks([], [])
    plt.title('Confusion matrix ')
    plt.colorbar()
    plt.savefig(pathsave + filename + "_confusion_matrix2.png")
    plt.show()
    plt.close()
    
def _draw_confusion_metrics__(y_true=None, y_pred=None, filename=None, pathsave=None):
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.savefig(pathsave + filename + "_confusion_matrix1.png")
    plt.show()
    plt.close()
    
    cm = confusion_matrix(y_true, y_pred)
    # Creating a dataframe for a array-formatted Confusion matrix,so it will be easy for plotting.
    cm_df = pd.DataFrame(cm)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm_df, annot=True)
    plt.title('Confusion Matrix')
    plt.ylabel('Actal Values')
    plt.xlabel('Predicted Values')
    plt.savefig(pathsave + filename + "_confusion_matrix3.png")
    plt.show()
    plt.close()
    return None
