# -*- coding: utf-8 -*-
"""
Created on Tue May 26 22:56:19 2020

@author: Kaush
"""



import pandas as pd
import numpy as np
import spacy as sp
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import re

from sklearn.preprocessing import LabelEncoder
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb






data_train = pd.read_csv(r"C:\Users\Kaush\Downloads\HackerEarth Data_set\MothersDay\train.csv")
data_test = pd.read_csv(r"C:\Users\Kaush\Downloads\HackerEarth Data_set\MothersDay\test.csv")

# print(data_train.info())
# print("Test info : \n ", data_test.info())


for i in range(len(data_train.id)):
    if(type(data_train.retweet_count[i])==float):
        # print(i)
        data_train.retweet_count[i] = "0"

for i in range(len(data_train.id)):
    if(data_train.retweet_count[i].isnumeric()):
        # print(i)
        data_train.retweet_count[i] = int(data_train.retweet_count[i])
    else:
        if(data_train.original_author[i].isnumeric()):
            data_train.retweet_count[i] = data_train.original_author[i]
        else:
            data_train.retweet_count[i] = 0
            

        
        
# data_train.info()

data_train.retweet_count.astype('int16')
# print((data_train.retweet_count).dtype)
# data_train.info()



X = pd.DataFrame(columns = ['original_text', "retweet_count"])
X['original_text'] = data_train['original_text'].copy
X["retweet_count"] = data_train["retweet_count"].copy
y = data_train['sentiment_class']
nlp = sp.load("en_core_web_sm")




def processing(txt):
    txt.lower()
    txt = re.sub(r"(pic.twitter.com.*)", "", txt)
    txt = re.sub(r"(https:.*)", "", txt)
    txt = re.sub("(#+)", "", txt)
    txt = re.sub("(@+)", "", txt)
    # txt = re.sub("$+", "", txt)
    # txt = re.sub("%+", "", txt)
    txt = re.sub("&+", "", txt)
    txt = re.sub("\s+", " ", txt)
#     txt = re.sub(" ", " ", txt)
#     print(txt)
    obj = nlp(txt)
    txt1 = [wd.text for wd in obj if wd.is_stop != True and wd.pos_ != 'PUNCT' ]
    txt1 = " ".join(txt1)
    txt1 = re.sub("\s+", " ", txt1)
#     print(txt1)
    return(txt1)
    

print("################# removing unwanted text #####################")

data_train.original_text = data_train.original_text.apply(processing)
data_train.drop(["id", "lang", "original_author", "sentiment_class"], axis =1, inplace=True)


#################################### Dont got above #####################################################

print("############## Spliting train and test data ##############")

X_train, X_test, y_train, y_test = train_test_split(data_train, y, test_size = 0.3, random_state= 10)


tf = TfidfVectorizer()
a = tf.fit_transform(X_train.original_text)
XX_train = tf.transform(X_train.original_text)
XX_test = tf.transform(X_test.original_text)

X = XX_train.toarray()
XX = XX_test.toarray()

# print(tf.get_feature_names())


# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import f1_score

# nb = GaussianNB(   )
# nb.fit(X.toarray(), y_train)
# y_pred = nb.predict(X_test)
# print(f1_score(y_pred, y_test, average='weighted'))


lgb_params = {
    "objective" : "classification",
    "metric" : "f1_score",
    "boosting": 'gbdt',
    "max_depth" : -1,
    "num_leaves" : 13,
    "learning_rate" : 0.1,
    "min_data_in_leaf": 0,
    "bagging_seed" : 1234,
    "verbosity" : 1,
    "lambda_l2" : 5.0,
    "seed": 1234,
    "drop_rate" : 0.5,
    "min_data_per_group":1,
    "max_bin" : 10,
}

print("################## Traaining Model #######################")
kf = KFold(n_splits=15,shuffle=True,random_state=31)

all_preds = []

for t,v in kf.split(X,y_train):
    print(t)
    x_train,x_valid,y_train,y_valid = train_test_split(X, y_train, test_size=0.3, random_state=t[0])
    lgb_train = lgb.Dataset(x_train,
                            label=y_train,
                            categorical_feature=['Subtopic','Description','Sex','Race','Grade','StratID1','StratID2','StratID3','StratificationType','YEAR','LocationDesc'])
    lgb_valid = lgb.Dataset(x_valid,
                            label=y_valid, 
                            categorical_feature=['Subtopic','Description','Sex','Race','Grade','StratID1','StratID2','StratID3','StratificationType','YEAR','LocationDesc'])
    evals_result = {}
    lgb_clf = lgb.train(lgb_params,
                    lgb_train,
                    100000,
                    valid_sets = [lgb_train,lgb_valid],
                    early_stopping_rounds=3000,
                    verbose_eval = 1000,
                    evals_result=evals_result
                   )
    all_preds.append(lgb_clf.predict(XX))
    
