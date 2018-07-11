import json
import numpy as np
import os, glob

from keras import objectives
from keras import metrics

from keras.models import Model

from keras.layers import Input, Dense
from keras.layers import Dropout, BatchNormalization
from keras.layers import RepeatVector
from keras.layers import Embedding

from keras.layers import Lambda
from keras.layers import LeakyReLU

from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional

from keras.utils.training_utils import multi_gpu_model
from keras.utils.vis_utils import plot_model
from keras.utils import np_utils

from keras.callbacks import ModelCheckpoint, EarlyStopping

import keras.backend as K

model_name = "Adversarial_autoencoder"

class ModelMGPU(Model):
    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)
    
class AAE() :
    # gensim에서 훈련시킨 word2vec model을 저장함
    def __init__(self, reverse_word_dic_path, model_path, embedding_weight_path) :
        self.reverse_word_dic = {}
        self.model_path = model_path
        self.embedding_weight = np.load(embedding_weight_path)
        
        self.pictures_path = self.model_path + "/pictures"
        self.architectures_path = self.model_path + "/architecture"
        self.weights_path = self.model_path + "/weights"
        
        if (os.path.exists(reverse_word_dic_path)) :
            with open(reverse_word_dic_path, "r", encoding = 'utf-8') as fp :
                self.revser_word_dic = json.load(fp)
        else :
            print(reverse_word_dic_path)
            print("해당 파일이 존재하지 않습니다.")
            quit()

        if (os.path.exists(self.model_path)) :
            pass
        else :
            os.makedirs(self.model_path)
            os.makedirs(self.pictures_path)
            os.makedirs(self.architectures_path)
            os.makedirs(self.weights_path)
        
    def make_AAE(self, batch_size = 512, sentences_length = 300, embed_size = 100,
                 latent_size = 100, nb_classes = 6, nb_epoch = 100) :
        self.batch_size = batch_size
        self.sentences_length = sentences_length
        self.embed_size = embed_size
        self.latent_size = latent_size
        self.nb_classes = nb_classes
        self.nb_epoch = nb_epoch
        
        # encoder의 input
        original_sentences = Input(shape = (self.sentences_length, ), name = 'encoder_input')
        # encoder의 output
        latent_space = self.encoder_model(original_sentences)

        # autoencoder model
        reconstructed_sentences = self.decoder_model(latent_space)

        # discriminator model
        discriminator_input = Input(shape = (self.latent_size, ), name = 'discriminator_input')
        discriminator_decision = self.discriminator_model(discriminator_input)
        self.discriminator = Model(inputs = discriminator_input, outputs = discriminator_decision)

        print("discriminator model\n", self.discriminator.summary(), "\n\n\n")
        plot_model(self.discriminator, self.pictures_path + "/discriminator.png", show_shapes = True, show_layer_names = True)


        # adversarial_autoencoder 모델
        # generator를 훈련시킬 때는 discriminator의 weights를 고정함
        self.discriminator.trainable = False
        fake_or_real = self.discriminator(latent_space)

        # adversarial_autoencoder
        # encoder에서 나온 fake_or_real와 train함수의 ones와 비교
        self.adversarial_autoencoder = Model(inputs = original_sentences,
                                             outputs = [reconstructed_sentences, fake_or_real])
        
        print("adversarial_autoencoder model\n", self.adversarial_autoencoder.summary())
        plot_model(self.adversarial_autoencoder, self.pictures_path + "/adversarial_autoencoder.png",
                   show_shapes = True, show_layer_names = True)

        model = (self.discriminator).to_json()
        with open(self.architectures_path + "/discriminator.json", "w", encoding = 'utf-8') as fp :
            fp.write(model)

        model = (self.adversarial_autoencoder).to_json()
        with open(self.architectures_path + "/adversarial_autoencoder.json", "w", encoding = 'utf-8') as fp :
            fp.write(model)

            
    def encoder_model(self, encoder_input) :
        embedding_sentences = Embedding(self.embedding_weight.shape[0], self.embedding_weight.shape[1], weights = [self.embedding_weight],
                                        input_length = self.sentences_length, trainable = False,
                                        name = 'embedding_layer')(encoder_input)
        
        encoder_layer1 = Bidirectional(LSTM(100, return_sequences = True, dropout = 0.5, recurrent_dropout = 0.5),
                                       name = 'encoder_layer1', merge_mode = 'sum')(embedding_sentences)
        encoder_layer2 = Bidirectional(LSTM(50, return_sequences = True, dropout = 0.5, recurrent_dropout = 0.5),
                                       name = 'encoder_layer2', merge_mode = 'sum')(encoder_layer1)
        encoder_layer3 = Bidirectional(LSTM(25, return_sequences = False, dropout = 0.5, recurrent_dropout = 0.5),
                                       name = 'encoder_layer3', merge_mode = 'sum')(encoder_layer2)

        latent_mean = Dense(self.latent_size, name = 'latent_mean', activation = 'sigmoid')(encoder_layer3)
        latent_var = Dense(self.latent_size, name = 'latent_var', activation = 'sigmoid')(encoder_layer3)


        batch_size = self.batch_size; latent_size = self.latent_size
        
        def sampling(mean_and_var) :
            latent_mean, latent_var = mean_and_var
            batch_size = K.shape(latent_mean)[0]

            return latent_mean + K.random_normal(shape = (batch_size, latent_size)) * K.exp(latent_var / 2)
        
        return Lambda(sampling ,output_shape = (self.latent_size, ),
                      name = 'latent_space')([latent_mean, latent_var])

        
    def decoder_model(self, decoder_input) :
        repeater = RepeatVector(self.sentences_length, name = 'Repeater')(decoder_input)
        decoder_layer1 = Bidirectional(LSTM(25, return_sequences = True,
                                            dropout = 0.5, recurrent_dropout = 0.5),
                                       name = 'decoder_layer1', merge_mode = 'sum')(repeater)
        decoder_layer2 = Bidirectional(LSTM(50, return_sequences = True,
                                            dropout = 0.5, recurrent_dropout = 0.5),
                                       name = 'decoder_layer2', merge_mode = 'sum')(decoder_layer1)
        decoder_layer3 = Bidirectional(LSTM(100, return_sequences = True,
                                            dropout = 0.5, recurrent_dropout = 0.5),
                                       name = 'decoder_layer3', merge_mode = 'sum')(decoder_layer2)
        reconstructed_news = Bidirectional(LSTM(self.embed_size, return_sequences = True,
                                            dropout = 0.5, recurrent_dropout = 0.5),
                                       name = 'reconstructed_news', merge_mode = 'sum')(decoder_layer3)

        return reconstructed_news


    def discriminator_model(self, random_input) :
        dis_layer1 = Dense(512, name = 'discriminator_layer1')(random_input)
        layer1_activation = LeakyReLU()(dis_layer1)
        dis_layer2 = Dense(256, name = 'discriminator_layer2')(layer1_activation)
        layer2_activation = LeakyReLU()(dis_layer2)
        dis_output = Dense(1, activation = 'sigmoid',
                           name = 'discriminator_output')(layer2_activation)

        return dis_output
        

    def category_predictor_model(self, nb_classes) :
        pass


    # 이미 훈련한 모델을 재사용할 경우 load = True로 호출
    def train(self, train_data_path, test_data_path, load = False, disc_weights_path = 0, aee_weights_path = 0) :
        # 파일 불러오기
        train_X = []; train_Y = []
        
        if (os.path.exists(train_data_path)) :
            with open(train_data_path, "r", encoding = 'utf-8') as fp :
                trainSets = json.load(fp)
                train_X = trainSets["X"]; train_Y = trainSets["Y"]
        if (os.path.exists(test_data_path)) :
            with open(test_data_path, "r", encoding = 'utf-8') as fp :
                testSets = json.load(fp)
                test_X.extend(testSets["X"]); test_Y.extend(testSets["Y"])

        train_X = np.array(train_X); train_Y = np.array(train_Y)

        zeros = np.zeros(size = (batch_size, 1))
        ones = np.ones(size = (batch_size, 1))
        
        if load : # 저장한 weights를 불러옴
            self.discriminator.load_weights(disc_weights_path)
            self.adversarial_autoencoder.load_weights(aee_weights_path)
            
        parallel_discriminator = ModelGPU(self.discriminator, gpus = 2)
        parallel_discriminator.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
                                       
        parallel_aae = ModelGPU(self.adversarial_autoencoder, gpus = 2)
        parallel_aae.compile(loss = ['mse', 'binary_crossentropy'], loss_weights = [0.999, 0.001], optimizer = 'Adam')
                
        for epoch in range(nb_epoch) :
            # 매번 random한 data로 훈련함
            index = np.random.randint(0, train_X.shape[0], batch_size)
            selected_news = train_X[index]

            # discriminator 훈련
            # 가짜 input과 진짜 input 생성
            fake_input = self.encoder.predict(selected_news)
            real_input = np.random.normal(size = (self.batch_size, self.latent_size))

            fake_loss = self.discriminator.train_on_batch(fake_input, zeros)
            real_loss = self.discriminator.train_on_batch(real_input, ones)

            discriminator_loss = 0.5 * np.add(real_loss, fake_loss)

            # generator 훈련
            aae_loss = self.adversarial_autoencoder.train_on_batch(selected_news,
                            [self.vectorized_words(selected_news), ones])

            # weights 저장
            self.discriminator.save_weights(self.weights_path + "epoch_{:d}".format(epoch) +
                                    "__disc-acc_{:.4f}".format(discriminator_loss[1]) +
                                    "__disc-loss_{:4f}_".format(discriminator_loss[0]) + ".hdf5")
            self.adversarial_autoencoder.save_weights(self.weights_path + "epoch_{:d}".format(epoch) +
                                    "__AAE-loss_{:.4f}".format(aae_loss[1]) +
                                    "__AEE-mse_{:4f}_".format(aee_loss[0]) + ".hdf5")


def main() :  
    word_dic_path = "./naver_news/training/word-dic.json"
    reverse_word_dic_path = "./naver_news/training/word-dic-reverse.json"
    train_data_path = "./naver_news/training/data/train_result.json"
    test_data_path = "./naver_news/training/data/test_result.json"
    embedding_weight_path = "./embedding_weights.npy"
    aae_model_path = "./AAE_model"

    NB_CLASSES = 6          # 6개의 카테고리
    BATCH_SIZE = 512
    SENTENCE_LENGTH = 300   # 한 뉴스는 최대 300개의 단어로 이루어져 있음
    EMBED_SIZE = 200
    LATENT_SIZE = 100
    NB_EPOCH = 500
        
    aae = AAE(reverse_word_dic_path, aae_model_path, embedding_weight_path)
    aae.make_AAE(BATCH_SIZE, SENTENCE_LENGTH, EMBED_SIZE, LATENT_SIZE, NB_CLASSES, NB_EPOCH)

    # aae.train(...)
    

if __name__ == "__main__" :
    main()
    
