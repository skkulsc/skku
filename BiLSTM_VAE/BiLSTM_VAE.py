import json
import numpy as np
import os, glob

from keras import objectives

from keras.models import Sequential, Model

from keras.layers import Input, Dense
from keras.layers import Dropout, BatchNormalization
from keras.layers import RepeatVector

from keras.layers import Lambda

from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional

from keras.utils.training_utils import multi_gpu_model
from keras.utils.vis_utils import plot_model

from keras.callbacks import ModelCheckpoint, EarlyStopping

import keras.backend as K

model_name = "BiLSTM_VAE"

NB_CLASSES = 6          # 6개의 카테고리
BATCH_SIZE = 512
SEQUENCE_LENGTH = 300   # 한 뉴스는 최대 300개의 단어로 이루어져 있음
EMBED_SIZE = 100
LATENT_SIZE = 50
NB_EPOCH = 100

# args : [z_mean, z_log_var]
def sampling(args) :
    z_mean, z_log_var = args
    batch_size = K.shape(z_mean)[0]

    # mean : 0이고 std : 0.01인 정규분포 생성
    epsilon = K.random_normal(shape = (batch_size, LATENT_SIZE),
                            mean = 0.0, stddev = 0.01)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# autoencoder에 대한 loss function
# vae_loss : 실제 label인 X와 decoder에서 나온 값인 X'의 loss 측정 
def vae_loss(x, x_decoded_mean) :
    x = K.flatten(x)
    x_decoded_mean = K.flatten(x_decoded_mean)
    xent_loss = SEQUENCE_LENGTH * objectives.mean_squared_error(x, x_decoded_mean)
    kl_loss = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis = -1)

    return xent_loss + kl_loss

# 한 뉴스는 최대 300단어로 이루어져 있고, 한 단어는 사전 훈련된 100차원의 벡터로 이루어짐
# (BATCH_SIZE, SEQUENCE_LENGTH, EMBED_SIZE)
embedded_sequences = Input(shape = (SEQUENCE_LENGTH, EMBED_SIZE),
                                name = 'embedded_sequences')

# encoder
encoder_layer1 = Bidirectional(LSTM(80, return_sequences = True, dropout = 0.5, recurrent_dropout = 0.5),
                                name = 'encoder_layer1', merge_mode = 'sum')(embedded_sequences)
encoder_layer2 = Bidirectional(LSTM(60, return_sequences = True, dropout = 0.5, recurrent_dropout = 0.5),
                                name = 'encoder_layer2', merge_mode = 'sum')(encoder_layer1)
encoder_layer3 = Bidirectional(LSTM(40, return_sequences = False, dropout = 0.5, recurrent_dropout = 0.5),
                                name = 'encoder_layer3', merge_mode = 'sum')(encoder_layer2)

# z_mean과 z_log_var는 각각 latent_size만큼의 길이를 가짐
z_mean = Dense(LATENT_SIZE, name = 'z_mean', activation = 'sigmoid')(encoder_layer3)
z_log_var = Dense(LATENT_SIZE, name = 'z_log_var', activation = 'sigmoid')(encoder_layer3)

# vae_loss 함수
VAE_Loss = vae_loss

# sampling 함수의 input : [z_mean, z_log_var]
# sampling 함수의 output : z_mean + K.exp(z_log_var / 2) * epsilon
encoding_output = Lambda(lambda x : sampling(x), name = 'encoded_lambda')([z_mean, z_log_var])

# latent space를 이용한 news category classification model을 만듬
category_predictor_output = Dense(NB_CLASSES, activation = 'softmax', name = 'predictorOutput')(encoding_output)

# (batch_size, latent_space)의 2-dim shape의 encoded를
# (batch_size, SEQUENCE_LENGTH, latent_space)의 3-dim shape로 바꿔줌
repeater = RepeatVector(SEQUENCE_LENGTH, name = 'repeater')(encoding_output)

# decoder
decoder_layer2 = Bidirectional(LSTM(40, return_sequences = True, dropout = 0.5, recurrent_dropout = 0.5),
                        merge_mode = "sum", name = 'decoderLayer1')(repeater)
decoder_layer3 = Bidirectional(LSTM(60, return_sequences = True, dropout = 0.5, recurrent_dropout = 0.5),
                        merge_mode = "sum", name = 'decoderLayer2')(decoder_layer2)
decoder_layer4 = Bidirectional(LSTM(80, return_sequences = True, dropout = 0.5, recurrent_dropout = 0.5),
                        merge_mode = "sum", name = 'decoderLayer3')(decoder_layer3)
decoder_output = Bidirectional(LSTM(EMBED_SIZE, return_sequences = True, dropout = 0.5, recurrent_dropout = 0.5),
                        merge_mode = "sum", name = 'decoderLayer4')(decoder_layer4)

# autoencoder model
# input : embedding된 문장
# output : 예상 카테고리, 복원된 문장
VAE = Model(inputs = embedded_sequences, outputs = [decoder_output, category_predictor_output])

print(VAE.summary())
plot_model(VAE, './BiLSTM_VAE.png', show_shapes = True, show_layer_names = True)

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

# 체크포인트
model_path = "./models/" + model_name
if (os.path.exists(model_path)) :
    pass
else :
    os.makedirs(model_path)
    
model_weights_path = model_path + "/weights"
if (os.path.exists(model_weights_path)) :
    pass
else :
    os.makedirs(model_weights_path)
    
file_path = model_weights_path +  "/weights-improvement-{epoch:d}-{val_decoderLayer4_loss:.4f}-{val_predictorOutput_acc:.4f}.hdf5"

checkpoint = ModelCheckpoint(file_path, verbose = 1, save_best_only = False)
callbacks_list = [checkpoint]
print("checkpoint 설정")

# model 저장
model_json = VAE.to_json()
with open(model_path + "/" + model_name + ".json", "w") as fp :
    fp.write(model_json)
print("모델 저장")

# decoder의 loss function으로는 vae_loss를 사용
# category predictor의 loss function으로는 categorical_crossentropy를 사용
VAE.compile(optimizer = 'Adam',
        loss = [vae_loss, 'categorical_crossentropy'],
        metrics = ['accuracy'])

numpy_trainX = np.load("./naver_news/training/words_100/data/trainX_vector_result.npy")
numpy_trainY = np.load("./naver_news/training/words_100/data/trainY_vector_result.npy")

numpy_testX = np.load("./naver_news/training/words_100/data/testX_vector_result.npy")
numpy_testY = np.load("./naver_news/training/words_100/data/testY_vector_result.npy")

# multi gpu 사용
parallel_VAE = ModelMGPU(VAE, gpus = 2)

parallel_VAE.compile(optimizer = 'Adam',
                            loss = [VAE_Loss, 'categorical_crossentropy'],
                            metrics = ['accuracy'])

parallel_VAE.fit(numpy_trainX, y = [numpy_trainX, numpy_trainY],
                      batch_size = BATCH_SIZE, epochs = NB_EPOCH,
                      validation_data = (numpy_testX, [numpy_testX, numpy_testY]),
                      callbacks = callbacks_list)
