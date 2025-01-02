# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 10:42:58 2021

@author: Oyelade
"""
from utils.MeasureUtil import MeasureClassification
from utils.IOUtil import save_results_to_csv, save_solutions_to_csv, check_make_dir 
from utils.GraphUtil import _draw_rates__, _plot_training_result__, _draw_confusion_metrics__, _draw_score_confusion_matrix__
from utils.paper4plotpropagation import _draw_propagation_rates__
import numpy as np
import statistics



class ProcessResult:
	
    def __init__(self, params=None):
        self.log_filename='optimzed_result_file'
        self.epoch=params["cnn_epoch"]
        self.experiment = params["experiment"]
        self.log_filename = params["filename"]
        self.save_multimodal_results_dir=params["save_multimodal_results_dir"]
        self.colors=['orange', 'yellow', 'red', 'gray', 'brown', 'green', 'pink', 'cyan', 'maroon', 'blue', 'olive', 'purple']
        #check if this directory exist otherwise create it
        check_make_dir(self.save_multimodal_results_dir)
        
    def save_results(self, train_outcome, prediction):
        loss_train, accuracy_train, loss_eval, accuracy_eval, time_total_train=train_outcome
        time_predict, avg_pred, y_true, y_pred, x_test=prediction
        y_true=np.argmax(y_true, axis=1)
        y_pred=np.argmax(y_pred, axis=1)
        _plot_training_result__(loss_train, loss_eval, accuracy_train, accuracy_eval, self.epoch, self.experiment, self.save_multimodal_results_dir)
        measure = MeasureClassification(y_true=y_true, y_pred=y_pred, number_rounding=4)
        measure._fit__()
        result = {
            'experiment': self.experiment, 
            'loss_train':loss_train,
            'accuracy_train': accuracy_train, 
            'val_loss':loss_eval, 
            'val_acc': accuracy_eval,
            'time_total_train': time_total_train, 
            'time_predict': time_predict, 
            'score_confusion_matrix': measure.score_confusion_matrix,
            'score_classification_report': measure.score_classification_report,
            'score_accuracy':measure.score_accuracy,
            'score_precision':measure.score_precision,
            'score_recall': measure.score_recall,
            'score_f1':measure.score_f1,
            'cohen_kappa': measure.cohen_kappa,
            'sensitivity':measure.sensitivity,
            'specificity': measure.specificity,
            'score_matthews_corrcoef':measure.score_matthews_corrcoef,
            'score_roc_auc': measure.score_roc_auc,
            'score_top_k_accuracy':measure.score_top_k_accuracy,
            'score_multilabel_confusion_matrix': measure.score_multilabel_confusion_matrix,
            'score_jaccard_score':measure.score_jaccard_score,
            'avg_pred': avg_pred
        }
        save_results_to_csv(result, self.log_filename, self.save_multimodal_results_dir)
        _draw_confusion_metrics__(y_true, y_pred, self.log_filename, self.save_multimodal_results_dir)
        _draw_score_confusion_matrix__(measure.score_confusion_matrix, self.log_filename, self.save_multimodal_results_dir)
                    
    def _save_classifiers_results__(self, method, results, allfit, allcost, testAcc, featCnt, gbest):
        for algorithm in results:
            solution = {
                    'optimizer':method,
                    'classifer': algorithm[0], 
                    'accuracy': algorithm[1],
                    'precision': algorithm[2],
                    'recall':algorithm[3],
                    'f1':algorithm[4],
                    'classifcation_report': algorithm[5],
                    'confusion_matrix':algorithm[6],
                    'testAccKNN':testAcc['knn'][0],
                    'testAccMLP':testAcc['mlp'][0],
                    'testAccDT':testAcc['dt'][0],
                    'testAccRF':testAcc['rf'][0],
                    'FeatureCount':featCnt,
                    'fitness':allfit,
                    'avgfitness':statistics.mean(allfit),
                    'best':gbest,
                    }
            save_results_to_csv(solution, 'optimization_classifier_results.csv', self.save_multimodal_results_dir)
        #_draw_confusion_metrics__(y_true, y_pred, self.log_filename, self.path_save_result)
        #_draw_score_confusion_matrix__(measure.score_confusion_matrix, self.log_filename, self.path_save_result)
        
    def save_probs_optimized_solutions(self, solution, fitness, probs, histoprobs, mammoprobs,
                                       histpredlabel, histotruelabel, mammopredlabel, mammotruelabel,
                                       histosolution, mammosolution, histonewlabel, mammonewlabel):
        item = {
                    'solution':solution,
                    'fitness': fitness, 
                    'probs': probs,
                    'histoprobs': histoprobs,
                    'mammoprobs':mammoprobs,
                    'histpredlabel':histpredlabel,
                    'histotruelabel': histotruelabel,
                    'mammopredlabel':mammopredlabel,
                    'mammotruelabel':mammotruelabel,
                    'histosolution':histosolution,
                    'mammosolution':mammosolution,
                    'histonewlabel':histonewlabel,
                    'mammonewlabel':mammonewlabel,
                    }
        save_results_to_csv(item, 'probs_optimized_results.csv', self.save_multimodal_results_dir)
        