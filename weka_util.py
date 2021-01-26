import weka.core.jvm as jvm
jvm.start(system_cp=True, packages=True)

from weka.attribute_selection import ASSearch, ASEvaluation, AttributeSelection
from weka.core.dataset import create_instances_from_lists
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
#import warnings
#warnings.filterwarnings("ignore")

def find_features(weka_df):
    search = ASSearch(classname="weka.attributeSelection.BestFirst")
    evaluator = ASEvaluation(classname="weka.attributeSelection.CfsSubsetEval")
    attsel = AttributeSelection()
    attsel.search(search)
    attsel.evaluator(evaluator)
    attsel.select_attributes(weka_df)
    print("# attributes: " + str(attsel.number_attributes_selected))
    print("attributes: " + str(attsel.selected_attributes))
    print("result string:\n" + attsel.results_string)

    return attsel.number_attributes_selected, attsel.selected_attributes, attsel.results_string

def plot_confusion(df, save_path):
    model_name = "HC_AD"
    target = df["label"]
    pred = df["prediction"]
    name = model_name.split("_")
    tn, fp, fn, tp = confusion_matrix(target, pred).ravel()
    stats = np.array([[tp, fn], [fp, tn]])
        
    plt_cm_1 = pd.DataFrame(stats, columns = [name[1] +"(Pos)", name[0] +"(Neg)"],
                                index = [name[1]+"(Pos)", name[0]+"(Neg)"])
    sns.heatmap(plt_cm_1, annot = True, cmap="YlGnBu", fmt = "g")
    
    plt.title("CM: HC vs. AD")
    plt.xlabel("True label")
    plt.ylabel("Predicted label")
    plt.savefig(os.path.join(save_path))
    
def plot_roc_segment(model, test_X, test_y, save_path):
    metrics.plot_roc_curve(model, test_X,  test_y, name = "HC vs. AD")
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Chance', alpha=.8)
    plt.title("ROC curve(Segment)")
    plt.savefig(os.path.join(save_path))
    
def plot_roc_person(df, save_path):
    lw =2
    fpr, tpr, thres = roc_curve(df["label"].values, 
                                (df["probability"].apply(lambda x: x[1])).values, pos_label = 1)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkblue',
             label='HC vs. AD (AUC = %0.2f)' % metrics.auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], color='red', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve(Person)')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_path))
    
def plot_roc_person_fromSeg(df, save_path):
    fpr, tpr, thres = roc_curve(df["label"].values, 
                                (df["num_ad"] / df["records"]).values, pos_label = 1)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkblue',
             label='HC vs. AD (AUC = %0.2f)' % metrics.auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve(Person)')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_path))