import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from .matrix_factorization import *
from polls.models import UserNewsTable, AuthUser
from sqlalchemy import create_engine
from .Config import DB_config

class CF() :
    def __init__(self) :
        self.db_config = DB_config()
        self.engine = create_engine(self.db_config.address)

    # 일정 변화가 있을 경우, update_info를 실행해 matrix factorization의 input을 upate해줌
    def update_count_table(self) :
        # 원본 테이블
        self.full_table = pd.read_sql(sql = 'select user_id, news_id, count from user_news_table', con = self.engine) 
                
        # MF를 하기 전 수정되는 테이블
        self.count_table = pd.read_sql(sql = 'select user_id, news_id, count from user_news_table', con = self.engine) 
        
        self.num_user, self.user_dic = self.make_ID_dic(self.count_table, 'user_id')
        self.num_news, self.news_dic = self.make_ID_dic(self.count_table, 'news_id')
        
        # 읽은 뉴스가 5개 미만인 user는 MF에서 제외함.
        ID_list = list(self.user_dic.keys())
        for user_id in ID_list :
            read_news_idx = self.count_table.index[self.count_table['user_id'] == user_id].tolist()
            if len(read_news_idx) < 5 :
                self.count_table = self.count_table.drop(read_news_idx)

        # 어느 유저와도 접점이 없는 user는 MF에서 제외함.
        self.common_dic = {}
        ID_list = list(self.user_dic.keys())
        for user_id in ID_list :
            news_idx = list(self.count_table.loc[self.count_table['user_id'] == user_id]['news_id'].values)

            common_nums = []
            for other_user_id in ID_list :
                if user_id == other_user_id :
                    continue
                else :
                    other_news_idx = list(self.count_table.loc[self.count_table['user_id'] == other_user_id]['news_id'].values)
                    length = len(set(news_idx).intersection(other_news_idx))
                    common_nums.append(length)

            if sum(common_nums) == 0 : # 공통된 news가 하나도 없을 때
                news_idx = self.count_table.index[self.count_table['user_id'] == user_id].tolist()
                self.count_table = self.count_table.drop(news_idx)

            self.common_dic[user_id] = sum(common_nums)

        self.num_user, self.user_dic = self.make_ID_dic(self.count_table, 'user_id')
        self.num_news, self.news_dic = self.make_ID_dic(self.count_table, 'news_id')   
        
        train_size = len(self.count_table)
        full_size = self.num_user * self.num_news

        # count가 1보다 큰 row들을 모두 training data에 넣음
        idx_list = []
        for idx, row in self.count_table.iterrows() :
            if (row['count'] > 1) :
                idx_list.append(idx)

        self.more_2_table = self.count_table.loc[idx_list]

        self.count_table = self.count_table.drop(idx_list)

        # 읽은 뉴스가 5개 미만인 user들의 data를 모두 training data에 넣음
        ID_list = list(self.user_dic.keys())
        for user_id in ID_list :
            read_news_idx = self.count_table.index[self.count_table['user_id'] == user_id].tolist()
            if len(read_news_idx) < 5 :                
                self.more_2_table = self.more_2_table.append(self.count_table.loc[read_news_idx])
                self.count_table = self.count_table.drop(read_news_idx)

        total_length = self.user_dic
    def update_info(self) :
        self.update_count_table()       

        # 각 유저마다 최소 몇 개의 뉴스를 읽었는지 확인 후, n_splits 값을 min값으로 정함
        min_ = 10
        for user_id in list(self.user_dic.keys()) :
            length = len(self.count_table.loc[self.count_table['user_id'] == user_id])
            if (length >= 2 and min_ > length) :
                min_ = length
        
        # 최소 5등분, 최대 10등분
        self.n_splits = min(min_, 10) 
        

    def make_ID_dic(self, table, column_name, start_ID = 0) :
        elements = table[column_name]
        num_elements = len(np.unique(elements))
    
        total_elements = np.unique([int(x) for x in elements])
        total_elements.sort()
    
        dic = {}
        for element in total_elements :
            if element not in dic.keys() :
                dic[element] = start_ID
                start_ID += 1
            
        return num_elements, dic

    def build_model(self, iters = 30, rank = 200, lr = 0.001, lr_decay = 0.96, reg = 0.001, step = 10, opt = 'adam')   :
        self.update_info()
        self.mf = MF(self.num_user, self.num_news, self.user_dic, self.news_dic, self.count_table, self.more_2_table, self.n_splits,
                 rank, lr, lr_decay, reg, step , opt)

        self.iters = iters

    # user가 읽지 않은 news에 대해서 rating을 기준으로 정렬해서 반환    
    def top_n_users(self) :
        self.mf.train_model(self.iters)

        best_rmse = self.mf.best_rmse_history
        outliers = [outlier for outlier in best_rmse if outlier > (np.mean(best_rmse) + 1.5 * np.std(best_rmse))]

        learnt_W, learnt_H, learnt_W_bias, learnt_H_bias, pred_matrix = self.mf.return_predMatrix()

        # 이미 읽은 news는 추천해주지 않음
        already_read = {}
        user_id_set = set(pd.read_sql("select user_id from user_news_table", con = self.engine)['user_id'])
        for user_id in user_id_set :
            temp = pd.read_sql("select news_id from user_news_table where user_id = {}".format(user_id), con = self.engine)['news_id'].values
            already_read[user_id] = temp

        top_n_dic = {}

        # MF에 사용된 news들에 대해서 추천을 해줌
        target_newsIDs = list(set(pd.read_sql(sql = 'select news_id from user_news_table', con = self.engine)['news_id'].values))
        for user_Id in list(self.user_dic.keys()) :
            top_n_dic[user_Id] = []
            temp = 0
            for news_Id in target_newsIDs :
                # 만약 해당 user가 이 뉴스를 읽지 않았다면
                try :
                    if news_Id not in already_read[user_Id] :
                        pred_counts = pred_matrix[self.user_dic[user_Id]][self.news_dic[news_Id]]
                        temp += pred_counts

                except Exception as e :
                    pass

            temp = np.mean(pred_counts)
            #print("user_Id : ", user_Id, "   counts mean : ", temp)

            for news_Id in target_newsIDs :
                try :
                    if news_Id not in already_read[user_Id] :
                        pred_counts = pred_matrix[self.user_dic[user_Id]][self.news_dic[news_Id]]
                        if pred_counts > temp :
                            top_n_dic[user_Id].append([news_Id, pred_counts])
                except Exception as e :
                    pass

        # 각 user당 ratings으로 정렬
        for key in list(top_n_dic.keys()) :
            top_n_dic[key] = sorted(top_n_dic[key], key = operator.itemgetter(1), reverse = True)

        # userId 별로 정렬
        final_n_dic = dict(sorted(top_n_dic.items(), key = operator.itemgetter(0), reverse = False))

        print("self.common_dic:\n{}\n".format(self.common_dic))
        
        return final_n_dic, self.common_dic, len(outliers)