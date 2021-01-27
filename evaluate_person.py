import pandas as pd
import os
import sys
sys.path.append("./src")
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from util import *
from weka_util import *
np.set_printoptions(suppress=True)
import json

if __name__ == "__main__":
    with open("./config/evaluate_person.json") as param:
            all_info = json.load(param)
            test_data = all_info["test_data"]
            train_data = all_info["train_data"]
            model_param = all_info["model_param"]
            saving_result_path = all_info["saving_result_path"]
            to_AutoML = all_info["to_AutoML"]

    param.close()

    #get_MFCC(**process_data)
    print("param loaded")
    
    #------------------------------------------load training data----------------------------------------------------

    file = open(to_AutoML["featName_AutoML"], 'r')
    param = list(json.load(file).keys())[1:]
    
    ct_path_train = train_data["control_csv"]
    ad_path_train = train_data["dementia_csv"]
    ct_df_train = pd.read_csv(ct_path_train)
    ad_df_train = pd.read_csv(ad_path_train)
    train_ad_ct = pd.concat([ct_df_train, ad_df_train]).reset_index(drop = True)
    
    file = open(to_AutoML["featName_AutoML"], 'r')
    param = list(json.load(file).keys())[1:]
    file.close()
    
    targets_ct = np.array([0] * len(ct_df_train))
    targets_ad = np.array([1] * len(ad_df_train))
    targets_ad_ct = np.hstack([targets_ct, targets_ad])
    train_data = train_ad_ct[param]
    std = StandardScaler()
    std.fit(train_data)
    X = std.transform(train_data)
    y = targets_ad_ct
    X, y = shuffle(X, y)
    print("training loaded")

    #------------------------------------------load testing data----------------------------------------------------

    ct_path_test = test_data["control_csv"]
    ad_path_test = test_data["dementia_csv"]
    
    ct_df_test = pd.read_csv(ct_path_test).drop(["Name", "label"], axis = 1)
    name_label_1 = pd.read_csv(ct_path_test)[["Name", "label"]]
    ad_df_test = pd.read_csv(ad_path_test).drop(["Name", "label"], axis = 1)
    name_label_2 = pd.read_csv(ad_path_test)[["Name", "label"]]
    name_label = pd.concat([name_label_1, name_label_2])
    
    test_ad_ct = pd.concat([ct_df_test, ad_df_test]).reset_index(drop = True)
    targets_ct = np.array([0] * len(ct_df_test))
    targets_ad = np.array([1] * len(ad_df_test))
    targets_ad_ct = np.hstack([targets_ct, targets_ad])
    test_data = test_ad_ct[param]
    X_test = std.transform(test_data)
    y_test = targets_ad_ct
    
    print("testing loaded")

    #------------------------------------------evaluation------------------------------------------------------------

    mod_file = open(model_param["model_path"], "rb")
    model = pickle.load(mod_file)
    mod_file.close()
    
    train_acc = model.score(X, y)

    name_label["prediction"] = model.predict(X_test).astype(int)
    name_label["probability"] = model.predict_proba(X_test).round(3).tolist()
    name_label["is_correct"] = name_label["label"] == name_label["prediction"]
    
    plot_confusion(name_label, saving_result_path["cm_path_person"])
    plot_roc_person(name_label, saving_result_path["ROC_path_person"])
    
    with open(saving_result_path["feature_text"], "w") as f:
        for i in test_ad_ct.columns:
            f.write(i)
            f.write("\n")
    f.close()
    
    with open(saving_result_path["Readme_text"], "w") as f:
        f.write("AutoML: {} \n".format(type(model)))
        f.write("best training accuracy is {} \n".format(train_acc))
        #f.write("best validation accuracy is 0.68 \n")
        f.write("best testing accuracy is {} \n".format(name_label["is_correct"].mean()))
        f.write("CT training data: {} \n".format(len(ct_df_train)))
        f.write("AD training data: {} \n".format(len(ad_df_train)))
        f.write("CT testing data: {} \n".format(len(ct_df_test)))
        f.write("AD testing data: {} \n".format(len(ad_df_test)))
    f.close()
    
    name_label = name_label.set_index("Name", drop = True)
    name_label.to_csv(saving_result_path["classification_result_person"], index = True)
    
    print("finished")

