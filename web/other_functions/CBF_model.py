import numpy as np
import pandas as pd
import operator

from math import *
from scipy.spatial import distance
from sqlalchemy import create_engine

from .Config import autoencoder_config, lsi_config, DB_config

class CBF() :
    def __init__(self, ae_config, l_config, db_config) :
        print("CBF 모델을 성공적으로 생성했습니다.")
        self.db_config = db_config
        self.get_DB_info()

    # 굳이 업데이트가 필요없는 DB들은 미리 불러와서 저장함
    def get_DB_info(self) :
        print("\nGet DB info from CBF class in CBF_model module\n")
        engine = create_engine(self.db_config.address)
        
        # 서로 동일한 제목의 news일 경우 추천에서 1개는 제외함
        self.news_Id_date_table = pd.read_sql(sql = 'select news_id, title, date from news_info_table', con = engine)
 
        latent_table = pd.read_sql(sql = 'select * from news_latent_space_table', con = engine)
        
        self.news_latent_dic = {}
        for i in range(len(latent_table)) :
            row = latent_table.iloc[i]
            self.news_latent_dic[row[0]] = np.asarray([float(value) for value in row[1].split()])
            
    # 각 news들의 latent space를 비교해 가장 유사한 주제를 가진 news를 반환
    # newsId = 읽은 news들의 ID로 이루어진 리스트
    def find_similar_news(self, newsId_list) :
        rec_news = {}
        for movie_Id in newsId_list :
            target_vector = self.news_latent_dic[movie_Id]
            
            dis_dic = {}
            for key in list(self.news_latent_dic.keys()) :
                dis_dic[key] = distance.euclidean(target_vector, self.news_latent_dic[key])
                
            dis_list = list(dis_dic.values())
            dis_list.sort(reverse = False)
            
            sorted_dis = sorted(dis_dic.items(), key = operator.itemgetter(1), reverse = False)
            idx = max(loc for loc, var in enumerate(dis_list) if var == 0.0)
            
            cutting = dis_list[idx + 1 : idx + 1 + 100]
            mean = np.mean(cutting); std = np.std(cutting)
            
            how_many = len([x for x in cutting if (x < mean - 2 * std)])
            
            Id_date_list = []
            for Id, _ in sorted_dis[idx + 1 : idx + 1 + how_many] :
                Id_date_list.append([Id, self.news_Id_date_table.loc[Id - 1]['date'], self.news_Id_date_table.loc[Id - 1]['title']]) 
                
            sorted_Id_date_list = sorted(Id_date_list, key = operator.itemgetter(1), reverse = True)
            
            rec_news[movie_Id] = sorted_Id_date_list
            
        return rec_news
        
    def recommend_based_cbf(self, newsId_list, nums) :
        print("newsId_list : ", newsId_list)
        rec_news = self.find_similar_news(newsId_list)

        prob_list = []
        for i in range(len(newsId_list)) :
            ratio = floor(i / 2) + 1
            prob_list.append(1 / (2 ** (ratio + 1)))
            
        prob_list.sort(reverse = False)
        prob_list = np.asarray(prob_list) + ((1 - sum(prob_list)) / len(prob_list))
        
        number = [0, ] * len(newsId_list)
        result = list(np.random.choice(len(newsId_list), size = nums, p = prob_list))
        for idx in result :
            number[idx] += 1
            
        key_list = list(rec_news.keys())

        final_list = []
        for i in range(len(number) - 1, -1, -1) :
            if number[i] == 0 :
                continue
                
            news_list = rec_news[key_list[i]][ : number[i]]
            sorted_list = sorted(news_list, key = operator.itemgetter(1), reverse = True)
            final_list.extend(sorted_list)

        return sorted(final_list, key = operator.itemgetter(1), reverse = True)
