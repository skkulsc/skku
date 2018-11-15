import numpy as np
import pandas as pd
import tensorflow as tf
import operator

from sklearn.model_selection import StratifiedKFold

class MF() :
    # pd.read_sql을 통해 pd.DataFrame type의 인스턴스를 얻은 뒤, 그대로 보내면 됨
    def __init__(self, num_users, num_news, user_dic, news_dic,  count_table, must_train_table, n_splits, 
                 rank = 200, lr = 0.01, lr_decay = 0.96, reg = 0.001, step = 10, opt = 'gd', print_tensor = False) :           
        print("rank = {}, lr = {}, lr_decay = {}, reg = {}, step = {}, opt = {}".format(
            rank, lr, lr_decay, reg, step, opt))
        
        self.count_table = count_table
        self.must_train_table = must_train_table
        
        self.opt = opt
        self.reg_value = reg
        
        self.num_users = num_users
        self.num_items = num_news
        print("self.num_users : ", self.num_users, " - self.num_items : ", self.num_items)
        
        self.user_dic = user_dic
        self.news_dic = news_dic

        self.global_mean = np.mean(self.count_table['count'].values)
        self.global_mean_tensor = tf.constant(self.global_mean, dtype = 'float32')
        
        # K-fold
        self.kf = StratifiedKFold(n_splits = n_splits, shuffle = True)
        self.groups = self.count_table.user_id.values
        self.kf.get_n_splits(self.count_table, self.groups)
        
        # trainset, testset
        self.dataset = list(self.kf.split(self.count_table, self.groups))
        
        self.rank = rank
        self.lr = lr
        self.step = step

        self.learning_rate = lr
        self.decay = lr_decay
        
        self.print_tensor = print_tensor

    def train_model(self, iters) :        
        learning_rate = tf.placeholder('float32')        
        userId_list = tf.placeholder('int32', [None], name = 'userId_list')
        itemId_list = tf.placeholder('int32', [None], name = 'itemId_list')
        valid_userId_list = tf.placeholder('int32', [None], name = 'valid_userId_list')
        valid_itemId_list = tf.placeholder('int32', [None], name = 'valid_itemId_list')
        rates_list = tf.placeholder('float32', [None], name = 'rates_list')
        valid_rates_list = tf.placeholder('float32', [None], name = 'valid_rates_list')
        
        num_rates = tf.placeholder('float32')
        val_num_rates = tf.placeholder('float32')
        
        print("Iters : ", iters)
        
        with tf.name_scope("Factorization") :
            W = tf.Variable(tf.truncated_normal([self.num_users, self.rank], stddev = 0.1), name = 'users') # user 
            W_bias = tf.Variable(tf.truncated_normal([self.num_users, 1], stddev = 0.1), name = "W_bias")   # user bias
            
            H = tf.Variable(tf.truncated_normal([self.num_items, self.rank], stddev = 0.1), name = 'items') # item
            H_bias = tf.Variable(tf.truncated_normal([self.num_items, 1], stddev = 0.1), name = "H_bias")   # item bias
        
            user_bias = tf.concat([W, W_bias], axis = 1)
            item_bias = tf.concat([H, H_bias], axis = 1)

            pred_pref = tf.matmul(user_bias, item_bias, transpose_b = True) # shape = [num_users, num_items]
            
            if self.print_tensor :
                print("user_bias:\n{}\n".format(user_bias))
                print("item_bias:\n{}\n".format(item_bias))
                print("pred_pref:\n{}\n".format(pred_pref))            

        with tf.name_scope("Flatten") :
            flatten_matrix = tf.reshape(pred_pref, [-1]) # 1차원으로 늘림
            
            if self.print_tensor :
                print("flatten_matrix:\n{}\n".format(flatten_matrix))
            
        with tf.name_scope("Gather") :
            # self.userId_list * tf.shape(pred_pref)[1] => User의 위치
            # self.itemId_list => Item의 위치
            get_results = tf.gather(flatten_matrix, userId_list * tf.shape(pred_pref)[1] + itemId_list,
                                    name = 'extracting_user_rating')
            get_results_val = tf.gather(flatten_matrix, valid_userId_list * tf.shape(pred_pref)[1] + valid_itemId_list,
                                        name = 'extracting_user_rating_val')
            
            if self.print_tensor :
                print("get_results:\n{}\n".format(get_results))
                print("get_results_val:\n{}\n".format(get_results_val))
                    
        with tf.name_scope("Subtract") :
            sub = tf.subtract(rates_list, get_results + self.global_mean_tensor)
            sub_val = tf.subtract(valid_rates_list, get_results_val + self.global_mean_tensor)
            
            if self.print_tensor :
                print("sub:\n{}\n".format(sub))
                        
        with tf.name_scope("Basic_Loss") :
            basic_loss = tf.sqrt(tf.reduce_sum(tf.pow(sub, 2)))
            basic_loss_val = tf.sqrt(tf.reduce_sum(tf.pow(sub_val, 2)))
            
            if self.print_tensor :
                print("cost:\n{}\n".format(basic_loss))
            
        with tf.name_scope("Regularization") :
            lda = tf.constant(self.reg_value, name = 'lambda', dtype = 'float32')
            norm_sums = tf.nn.l2_loss(W) + tf.nn.l2_loss(W_bias) + tf.nn.l2_loss(H) + tf.nn.l2_loss(H_bias)
            regul = tf.multiply(norm_sums, lda, name = 'regularizer')
            
        with tf.name_scope("Final_Loss") :
            final_loss = basic_loss + regul
            
            if self.print_tensor :
                print("final_loss :\n{}\n".format(final_loss))

        with tf.name_scope("Optimizer") :
            if (self.opt == 'adam') :
                if self.print_tensor :
                    print("adam optimizer")
                optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(final_loss)
                
            elif (self.opt == 'gd') :
                if self.print_tensor :
                    print("gradient descent optimizer")
                optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(final_loss)
        

        with tf.name_scope("RMSE") :
            rmse = tf.sqrt(tf.divide(tf.reduce_sum(tf.pow(sub, 2)), num_rates))
            rmse_val = tf.sqrt(tf.divide(tf.reduce_sum(tf.pow(sub_val, 2)), val_num_rates))

        # 각각의 fold에 대해서 iters만큼의 training을 한 뒤, validation loss가 가장 작은 model을 선택해야함
        
        self.final_rmse = np.inf # K개의 model중에서 가장 작은 값을 찾음
        self.best_results = {}
        self.test_table = 0
        self.best_rmse_history = []
        
        for idx, (train_idx, test_idx) in enumerate(self.dataset) :
            training_table = self.count_table.iloc[train_idx]
            training_table = training_table.append(self.must_train_table)
            
            training_userId = [x for x in training_table.user_id.values]
            training_movieId = [x for x in training_table.news_id.values]
            training_rates = training_table['count'].values
                
            test_table = self.count_table.iloc[test_idx]
            testing_userId = [x for x in test_table.user_id.values]
            testing_movieId = [x for x in test_table.news_id.values]
            testing_rates = test_table['count'].values
            
            # 새로운 data로 바뀔때마다 tf.gather에서 원하는 data를 추출하기위해 다시 설정해줘야함
            userId_list_ = []
            for userId in training_userId :
                userId_list_.append(self.user_dic[userId])
    
            itemId_list_ = []
            for newsId in training_movieId :
                itemId_list_.append(self.news_dic[newsId])   
    
            valid_userId_list_ = []
            for userId in testing_userId :
                valid_userId_list_.append(self.user_dic[userId])

            valid_itemId_list_ = []
            for newsId in testing_movieId :
                valid_itemId_list_.append(self.news_dic[newsId])
            
            # 한 fold에 대해서 학습을 함
            kf_rmse = []
            sess = tf.Session()          
            init = tf.global_variables_initializer()
            sess.run(init)  
        
            best_epoch = 0
            best_rmse = np.inf
            best_results = {}
            history_loss = []
            
            lr_ = self.learning_rate            
            for i in range(iters) :
                optimizer_, rmse_, rmse_val_, learnt_W, learnt_H, learnt_W_bias, learnt_H_bias =\
                sess.run([optimizer, rmse, rmse_val, W, H, W_bias, H_bias],
                             feed_dict = {
                                 learning_rate : lr_,
                                 userId_list : userId_list_,
                                 itemId_list : itemId_list_,
                                 valid_userId_list : valid_userId_list_,
                                 valid_itemId_list : valid_itemId_list_,
                                 rates_list : training_rates,
                                 valid_rates_list : testing_rates,
                                 num_rates : float(len(training_rates)),
                                 val_num_rates : float(len(testing_rates))
                             })

                # best_rmse보다 현재 rmse_val_가 더 작다면, best_rmse의 값을 바꿔줌
                if (best_rmse > rmse_val_) :    
                    best_epoch = 0
                    best_rmse = rmse_val_
                    best_results['learnt_W'] = learnt_W
                    best_results['learnt_H'] = learnt_H
                    best_results['learnt_W_bias'] = learnt_W_bias
                    best_results['learnt_H_bias'] = learnt_H_bias
                else :
                    history_loss.append(rmse_val_)

                
                if (i == 0) :
                    print("\n------> [{:3d}] 번째 Model Training".format(idx + 1))
                    
                #if ((i == iters - 1)) :
                #    print("Epoch = [{:3d}] Training RMSE = [{:11.6f}] Learning Rate = [{:11.9f}] Val RMSE = [{:11.6f}] Best Val RMSE = [{:11.6f}]".
                #          format(i, rmse_, lr_, rmse_val_, best_rmse)) 
                    
                if (len(history_loss) >= 3) :
                    # 현재 validation RMSE가 이전에 저장된 3개 이상의 validation RMSE의 가장 작은 값보다 작다면 learning rate을 줄임
                    if (min(history_loss[ : ]) >  best_rmse) :
                        lr_ = lr_ * self.decay
                        history_loss = []

            # self.final_rmse보다 현재 model의 best validation rmse가 더 작다면, update 시켜줌
            if (self.final_rmse > best_rmse) :
                self.final_rmse = best_rmse
                self.best_results['learnt_W'] = best_results['learnt_W']
                self.best_results['learnt_H'] = best_results['learnt_H']
                self.best_results['learnt_W_bias'] = best_results['learnt_W_bias']
                self.best_results['learnt_H_bias'] = best_results['learnt_H_bias']
                self.test_table = test_table
                
            self.best_rmse_history.append(best_rmse)
            #print("len(training_table) = [{:5d}] -- len(testing_table) = [{:5d}]".format(
            #    len(training_table), len(test_table)))
                  
        print("\nthe smallest validation RMSE : ", self.final_rmse)
        
        self.learnt_W = self.best_results['learnt_W']
        self.learnt_H = self.best_results['learnt_H']
        self.learnt_W_bias = self.best_results['learnt_W_bias']
        self.learnt_H_bias = self.best_results['learnt_H_bias']

    # training 결과에 대한 matrix를 반환함
    def return_predMatrix(self) :
        pred_matrix = np.dot(np.concatenate([self.learnt_W, self.learnt_W_bias], axis = 1), 
                             np.concatenate([self.learnt_H, self.learnt_H_bias], axis = 1).T) + self.global_mean
        return self.learnt_W, self.learnt_H, self.learnt_W_bias, self.learnt_H_bias, pred_matrix
        
    # test data에 대해 model이 예상하는 rating과 실제 rating을 비교함
    def compare_count(self) :
        pred_matrix = np.dot(np.concatenate([self.learnt_W, self.learnt_W_bias], axis = 1), 
                             np.concatenate([self.learnt_H, self.learnt_H_bias], axis = 1).T) + self.global_mean
        
        top_n_dic = {}
        for i in range(len(self.test_table)) :
            user_Id = int(self.test_table.iloc[i][0])
            news_Id = int(self.test_table.iloc[i][1])
            rating_ = self.test_table.iloc[i][2]
            
            if user_Id not in top_n_dic.keys() :
                top_n_dic[user_Id] = []
                
            top_n_dic[user_Id].append([news_Id, pred_matrix[self.user_dic[user_Id]][self.news_dic[news_Id]], rating_])
        
        # 각 user당 count으로 정렬
        for key in list(top_n_dic.keys()) :
            top_n_dic[key] = sorted(top_n_dic[key], key = operator.itemgetter(1), reverse = True)
            
        # userId 별로 정렬
        final_n_dic = dict(sorted(top_n_dic.items(), key = operator.itemgetter(0), reverse = False))
            
        return final_n_dic