from .Condition import condition
from .CF_model import CF
from .CBF_model import CBF
from .Config import *
import numpy as np
import pandas as pd
import time
import random

from django.db.models import Count

class Rec_system() :
    def __init__(self, NewsInfoTableObj, AuthUserObj, UserNewsTableObj, engine) :
        np.random.seed(int(time.time()))
        
        self.engine = engine
        self.NewsInfoTableObj = NewsInfoTableObj
        self.AuthUserObj = AuthUserObj
        self.UserNewsTableObj = UserNewsTableObj
        
        self.newsListTable = pd.read_sql(sql = "select * from news_info_table", con = self.engine)
           
        self.cf = CF()
        self.cbf = CBF(autoencoder_config(), lsi_config(), DB_config())
        
        self.top_k_users = None
        self.common_dic = None
        self.num_outlier = None
        self.CBF_dict = dict()
        
        # 모든 user들에 대하여 미리 CBF_dict()을 채움
        user_IDs = list(set(pd.read_sql(sql = 'select * from user_news_table', con = engine)['user_id']))
        print("user_IDs:\n{}\n".format(user_IDs))
        
        for user_ID in user_IDs :
            sql = "select * from user_news_table where user_id = {} order by read_time DESC".format(user_ID)  
            user_read_news = list(pd.read_sql(sql, con = engine)['news_id'])

            recent_read_news = user_read_news[ : 10] # 최근에 읽은 10개의 news를 가져옴
            recent_read_news.reverse() # 최근에 읽은 뉴스가 뒤로 오게

            final_list = self.cbf.recommend_based_cbf(recent_read_news, 100)        
            self.CBF_dict[user_ID] = final_list
            
    # DB table과 관련된 인스턴스 업데이트
    def update_table(self, NewsInfoTableObj = None, AuthUserObj = None, UserNewsTableObj = None) : 
        if (NewsInfoTableObj is not None) :
            self.NewsInfoTableObj = NewsInfoTableObj
            self.newsListTable = pd.read_sql(sql = "select * from news_info_table", con = self.engine)
            
        if (AuthUserObj is not None) :
            self.AuthUserObj = AuthUserObj
            
        if (UserNewsTableObj is not None) :
            if (UserNewsTableObj == "not") :
                # 가장 읽기있는 뉴스
                self.records_by_pop = self.UserNewsTableObj.values("news").annotate(score = Count('news')).order_by('-score') # 중복 클릭 허용 X
            else :
                self.UserNewsTableObj = UserNewsTableObj
                self.records_by_pop = self.UserNewsTableObj.values("news").annotate(score = Count('news')).order_by('-score') # 중복 클릭 허용 X

    # CBF를 수행하는 함수
    def do_CBF(self, userID) :                    
        sql = "select * from user_news_table where user_id = {} order by read_time DESC".format(userID)  
        user_read_news = list(pd.read_sql(sql, con = self.engine)['news_id'])
        
        recent_read_news = user_read_news[ : 10] # 최근에 읽은 10개의 news를 가져옴
        recent_read_news.reverse() # 최근에 읽은 뉴스가 뒤로 오게 

        final_list = self.cbf.recommend_based_cbf(recent_read_news, 100)        
        self.CBF_dict[userID] = final_list
        
    # matrix factorization을 수행하는 함수
    def do_MF(self, iters = 200, rank = 100, lr= 0.001, reg = 0.01, step = 10, lr_decay = 0.9, opt = 'adam') :
        try :
            if (condition(self.cbf.db_config.address)) :
                self.cf.build_model(iters, rank, lr, reg, step, lr_decay, opt = 'adam')
                self.top_k_users, self.common_dic, self.num_outlier = self.cf.top_n_users()
            else :
                self.top_k_users = None
                self.common_dic = None
                self.num_outlier = None         
                
        except Exception as e :
            print("\nError in do_MF --- Rec_system class")
            print("Error : \n{}\n".format(e))
            
            self.top_k_users = None
            self.common_dic = None
            self.num_outlier = None
    
    # news를 고름
    def extract_newsList(self) :
        np.random.seed(int(time.time()))
        random_idx = np.random.choice(len(self.newsListTable), 100, replace = False)
        randomNewsList = self.newsListTable.loc[random_idx]

        context = []
        
        for idx, newsInfo in randomNewsList.iterrows() :
            temp = dict()
            temp['news_id'] = newsInfo['news_id']
            temp['date'] = newsInfo['date']
            temp['category'] = newsInfo['category']
            temp['title'] = newsInfo['title']
            temp['content'] = newsInfo['content']
    
            context.append(temp)
        return context
    
    # newsListTable에서 해당 ID를 가진 news를 가져옴
    def get_news_from_ID(self, newsId_list) :
        if (len(newsId_list) == 0) :
            return []
        
        DB_list = []
        for ID in newsId_list :
            newsInfo = self.newsListTable.loc[self.newsListTable['news_id'] == ID]
    
            temp = dict()
            temp['news_id'] = newsInfo['news_id'].values[0]
            temp['date'] = newsInfo['date'].values[0]
            temp['category'] = newsInfo['category'].values[0]
            temp['title'] = newsInfo['title'].values[0]
            temp['content'] = newsInfo['content'].values[0]
    
            DB_list.append(temp)
        
        return DB_list
        
    def recommend(self, userID, newsId_list, context) :
        # 랜덤으로 고른 뉴스
        context['newsInfo'] = self.extract_newsList()
        
        total = 15 # 추천할 뉴스 개수
        recent_read_news = newsId_list[ : 10] # 최근에 읽은 10개의 news를 가져옴
        recent_read_news.reverse() # 최근에 읽은 뉴스가 뒤로 오게
        
        if (len(newsId_list) == 0) : # 읽은 뉴스가 하나도 없을 
            context['recent_news'] = []
            
            # 가장 조회수가 높은 뉴스들로 추천해줌
            records_by_pop = self.UserNewsTableObj.values("news").annotate(score = Count('news')).order_by('-score') # 중복 클릭 허용 X 
            popular_news_list = []
            for record in records_by_pop :
                if record['news'] not in newsId_list :
                    popular_news_list.append(record['news'])

                if (len(popular_news_list) == total) :
                    break

            context['rec_news'] = self.get_news_from_ID(popular_news_list)
            random.shuffle(context['rec_news'])
            
        else : # 읽은 뉴스가 1개 이상 
            context['recent_news'] = self.get_news_from_ID(recent_read_news) # 최근에 읽은 뉴스 저장

            cond = condition(self.cbf.db_config.address)            
            if (not cond and self.top_k_users is None) : # MF를 시행할 조건이 되지 않는다면
                # CBF를 먼저 적용함
                rec_news_list = []
                
                try :
                    final_list = self.CBF_dict[userID]
                    idx_list = np.random.choice(len(self.CBF_dict[userID]), min(15, len(self.CBF_dict[userID])), replace = False)
                    title_list = [] 
                    
                    for idx in idx_list :
                        ID, date, title = final_list[idx]
                        if (ID not in newsId_list) and (ID not in rec_news_list) and (title.strip() not in title_list) : # 중복방지
                            rec_news_list.append(ID)
                            title_list.append(title.strip())

                        if len(rec_news_list) == total :
                            break
                except Exception as e :
                    print("\nError in self.CBF_dict --- Rec_system class")
                    print("Error : \n{}\n".format(e))
                    
                print("userID : ", userID, " -> ", "CBF 통과 후 뉴스 개수 : ", len(rec_news_list))

                # 나머지는 조회수가 가장 높은 news로 채움
                if len(rec_news_list) < total :
                    # 가장 조회수가 높은 뉴스들로 추천해줌
                    records_by_pop = self.UserNewsTableObj.values("news").annotate(score = Count('news')).order_by('-score') # 중복 클릭 허용 X 
                    for record in records_by_pop :
                        if ((record['news'] not in newsId_list) and (record['news'] not in rec_news_list)) :
                            rec_news_list.append(record['news'])

                        if (len(rec_news_list) == total) :
                            break

                context['rec_news'] = self.get_news_from_ID(rec_news_list)
                random.shuffle(context['rec_news'])
                
            # 여기까지 구현함
            else : # MF를 시행할 조건이 된다면
                context['recent_news'] = self.get_news_from_ID(recent_read_news) # 최근에 읽은 뉴스 저장
                rec_news_list = []

                # MF
                try :
                    top_user_lists = self.top_k_users[userID] # userID에 대한 MF 결과를 가져옴
                    
                    if (self.num_outlier) :
                        top_user_lists = top_user_lists[ : int(len(top_user_lists) * (1 / (1 + self.num_outlier)))]
                    more_1_ID = [news_id for news_id, counts in top_user_lists]
                    range_ = len(more_1_ID) * float(self.common_dic[userID] / np.sum(list(self.common_dic.values())))
                    more_1_ID = more_1_ID[ : min(7, int(range_))]
                    
                    for ID in more_1_ID :
                        if ID not in newsId_list :
                            rec_news_list.append(ID)

                except Exception as e :
                    print("\nError in recommend --- more_1_ID --- Rec_system class")
                    print("Error : \n{}\n".format(e))

                print("userID : ", userID, " -> ", "MF 통과 후 뉴스 개수 : ", len(rec_news_list))
                
                # CBF
                if (len(rec_news_list) < total) :
                    
                    try :
                        final_list = self.CBF_dict[userID]
                        idx_list = np.random.choice(len(self.CBF_dict[userID]), min(15, len(self.CBF_dict[userID])), replace = False)
                        
                        title_list = [] 
                        for idx in idx_list :
                            ID, date,  title = final_list[idx]
                            if (ID not in newsId_list) and (ID not in rec_news_list) and (title.strip() not in title_list) : # 중복방지
                                rec_news_list.append(ID)
                                title_list.append(title.strip())

                            if len(rec_news_list) == total :
                                break  

                    except Exception as e :
                        print("\nError in self.CBF_dict --- Rec_system class")
                        print("Error : \n{}\n".format(e))
                        
                print("userID : ", userID, " -> ", "CBF 통과 후 뉴스 개수 : ", len(rec_news_list))
                
                # POP
                if (len(rec_news_list) < total) :
                    records_by_pop = self.UserNewsTableObj.values("news").annotate(score = Count('news')).order_by('-score') # 중복 클릭 허용 X

                    for record in records_by_pop :
                        if ((record['news'] not in newsId_list) and (record['news'] not in rec_news_list)) :
                            rec_news_list.append(record['news'])

                        if (len(rec_news_list) == total) :
                            break
                
                context['rec_news'] = self.get_news_from_ID(rec_news_list)
                random.shuffle(context['rec_news'])