from gensim.models.callbacks import CallbackAny2Vec

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
