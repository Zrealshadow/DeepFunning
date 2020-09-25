'''
 * @author Waldinsamkeit
 * @email Zenglz_pro@163.com
 * @create date 2020-09-25 15:32:18
 * @desc 
'''
from sklearn.metrics import confusion_matrix

'''---------------------- Binary confusion matrix evaluation -----------------'''
def binary_confusion_matrix_evaluate(y_true,y_pred):
    tn ,fp, fn, tp =  confusion_matrix(y_true,y_pred).ravel()
    acc = float(tn + tp)/(fp+fn+tn+tp) 
    prec =  float(tp) / (tp + fp) if (tp+fp) != 0 else 0.
    recall =  float(tp) / (tp + fn) if (tp + fn) != 0 else 0.
    f1= 2*prec*recall / ( prec + recall) if prec + recall !=0 else 0.
    return acc,prec,recall,f1