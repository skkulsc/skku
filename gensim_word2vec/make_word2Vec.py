import datetime

import boto
import json
import gensim
from gensim.models import Word2Vec
from gensim.test.utils import common_texts as sentences
from gensim.test.utils import get_tmpfile
import multiprocessing

class EpochSaver(CallbackAny2Vec):
     "Callback to save model after every epoch"
     def __init__(self, path_prefix):
         self.path_prefix = path_prefix
         self.epoch = 0
     def on_epoch_end(self, model):
         output_path = '{}_epoch{}.model'.format(self.path_prefix, self.epoch)
         print("Save model to {}".format(output_path))
         model.save(output_path)
         self.epoch += 1

class EpochLogger(CallbackAny2Vec):
     "Callback to log information about training"
     def __init__(self):
         self.epoch = 0
     def on_epoch_begin(self, model):
         print("Epoch #{} start at {}".format(self.epoch, datetime.datetime.now()))
     def on_epoch_end(self, model):
         print("Epoch #{} end at {}".format(self.epoch, datetime.datetime.now()))
         self.epoch += 1
        
epoch_saver = EpochSaver(get_tmpfile("temporary_model"))
epoch_logger = EpochLogger()

train_preprocessed_path = "./naver_news/training/words_100/data/train_reverse_result.json"
test_preprocessed_path = "./naver_news/training/words_100/data/test_reverse_result.json"

train_words_list = []
with open(train_preprocessed_path, "r", encoding = 'utf-8') as fp :
    data = json.load(fp)
    train_words_list = data["X"]
    
with open(test_preprocessed_path, "r", encoding = 'utf-8') as fp :
    data = json.load(fp)
    train_words_list.extend(data["X"])

model_name = "./naver_news/training/words_100/data/word2vecModel.model"

# embedding size = 100, skip-gram, 300 iterations
word2vec_model = Word2Vec(train_words_list, size = 100, window = 5, min_count = 1, sg = 1, iter = 300,
                             workers = multiprocessing.cpu_count(), 
                             callbacks=[epoch_saver, epoch_logger])
word2vec_model.save(model_name)
