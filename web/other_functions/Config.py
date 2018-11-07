from .config_info import *

class autoencoder_config() :
    def __init__(self) :        
        self.embedding_weight_path = embedding_weight_path
        self.ae_weight_path = ae_weight_path        
        self.word_dic_path = word_dic_path
        
        self.sequence_length = sequence_length
        self.word_embedding_size = word_embedding_size
        self.lstm_hidden_sizes = lstm_hidden_sizes
        
class lsi_config() :
    def __init__(self) :
        self.dictionary_path = dictionary_path
        self.tfidf_model_path = tfidf_model_path
        self.lsi_model_path = lsi_model_path
        
class DB_config() :
    def __init__(self) :
        self.address = DB_address
