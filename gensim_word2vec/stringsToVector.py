import gensim
from gensim.models import Word2Vec
from word2vec_callback import *

import json
import numpy as np
import os, glob
from copy import deepcopy
from keras.utils import np_utils

model = Word2Vec.load("./naver_news/training/words_100/word2vecModel.model")
word_vectors = model.wv
del model

data_path = "./naver_news/training/words_100/data/"
test_string_path = data_path + "test_reverse_result.json"
train_string_path = data_path + "train_reverse_result.json"
train_lists_X = []
train_lists_Y = []
test_lists_X = []
test_lists_Y = []

if (os.path.exists(data_path)) :
    temp = []
    
    with open(test_string_path, "r", encoding = 'utf-8') as fp :
        testSets = json.load(fp)
        
        Y_test = testSets["Y"]
        test_lists_Y = np_utils.to_categorical(Y_test, 6)
        del Y_test

        X_test = testSets["X"]; 
        print("X_test의 length : ", len(X_test))
        for newsContent in X_test :
            for word in newsContent :
                temp.append(word_vectors[word])
            test_lists_X.append(deepcopy(temp))
            temp.clear()
        
        del X_test

    np.save(data_path + "testX_vector_result.npy", test_lists_X)
    np.save(data_path + "testY_vector_result.npy", test_lists_Y)
    
    with open(train_string_path, "r", encoding = 'utf-8') as fp :
        trainSets = json.load(fp)
        
        Y_train = trainSets["Y"]
        train_lists_Y = np_utils.to_categorical(Y_train, 6)
        del Y_train

        X_train = trainSets["X"]; 
        print("X_train의 length : ", len(X_train))
        for newsContent in X_train :
            for word in newsContent :
                temp.append(word_vectors[word])
            train_lists_X.append(deepcopy(temp))
            temp.clear()

        del X_train

    np.save(data_path + "trainX_vector_result.npy", train_lists_X)
    np.save(data_path + "trainY_vector_result.npy", train_lists_Y)
