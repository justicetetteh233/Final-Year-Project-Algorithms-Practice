#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:33, 25/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error, r2_score
from numpy import round, sqrt, abs, where, mean#, asscalar
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score 
from sklearn.metrics import roc_auc_score, matthews_corrcoef,  multilabel_confusion_matrix, jaccard_score


"""
:param y_true:
:param y_pred:
:param multi_output:    string in [‘raw_values’, ‘uniform_average’, ‘variance_weighted’] or array-like of shape (n_outputs)
:param number_rounding:
"""
class MeasureClassification:
    def __init__(self, y_true=None, y_pred=None, number_rounding=3):
        self.y_true = y_true
        self.y_pred = y_pred
        self.number_rounding = number_rounding
        self.score_confusion_matrix, self.score_classification_report, self.score_accuracy, self.score_precision=None, None, None, None, 
        self.score_recall, self.score_f1, self.cohen_kappa, self.specificity, self.sensitivity= None, None, None, None, None
        self.score_matthews_corrcoef=None  
        self.score_roc_auc, self.score_top_k_accuracy=0.0, 0.0, 
        self.score_multilabel_confusion_matrix, self.score_jaccard_score=None, None
        
    def confusion_matrix(self):
        temp = confusion_matrix(self.y_true, self.y_pred)
        self.score_confusion_matrix=temp
    
    def classification_report(self):
        temp = classification_report(self.y_true, self.y_pred)
        self.score_classification_report=temp
    
    def accuracy_score(self):
        temp = accuracy_score(self.y_true, self.y_pred)
        self.score_accuracy=temp
    
    def precision_score(self):
        temp = precision_score(self.y_true, self.y_pred, average='micro')
        self.score_precision=temp
    
    def recall_score(self):
        temp = recall_score(self.y_true, self.y_pred, average='micro') #average=None, average='macro', weighted
        self.score_recall=temp
    
    def f1_score(self):
        temp = f1_score(self.y_true, self.y_pred, average='micro')
        self.score_f1=temp
    
    def cohen_kappa_score(self):
        temp = cohen_kappa_score(self.y_true, self.y_pred)
        self.cohen_kappa=temp
    
    def specificity_sensitivity(self):
        temp = confusion_matrix(self.y_true, self.y_pred)
        TN = temp[0][0]
        FN = temp[1][0]
        TP = temp[1][1]
        FP = temp[0][1]
        specificity = TN / (TN+FP)
        sensitivity  = TP / (TP+FN)
        self.specificity=specificity
        self.sensitivity=sensitivity    
    
    def roc_auc_score(self):
        temp = roc_auc_score(self.y_true_unnormalized, self.y_pred_unnormalized, multi_class="ovr",average='macro')
        self.score_roc_auc=temp
    
    def matthews_corrcoef(self):
        temp = matthews_corrcoef(self.y_true, self.y_pred, )
        self.score_matthews_corrcoef=temp
    '''
    def top_k_accuracy_score(self):
        temp = top_k_accuracy_score(self.y_true, self.y_pred, )
        self.score_top_k_accuracy=temp
    '''
    
    def multilabel_confusion_matrix(self):
        temp = multilabel_confusion_matrix(self.y_true, self.y_pred)
        self.score_multilabel_confusion_matrix=temp
    
    def jaccard_score(self):
        temp = jaccard_score(self.y_true, self.y_pred, average='micro')
        self.score_jaccard_score=temp
        
    def _fit__(self):
        self.confusion_matrix()
        self.classification_report()
        self.multilabel_confusion_matrix()
        self.accuracy_score()
        self.precision_score()
        self.recall_score()
        self.cohen_kappa_score()
        self.specificity_sensitivity()
        self.matthews_corrcoef()
        self.jaccard_score()
        self.f1_score()
    