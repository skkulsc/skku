import numpy as np
import pandas as pd
import time

from .ageGroup_calcultor import *
from .Config import *

class conf_matrix() :
    def __init__(self, AutoUserObj, UserNewsTableObj, UserScrapTableObj, engine) :
        self.engine = engine
        self.NewsInfoTableObj = NewsInfoTableObj
        self.AuthUserObj = AuthUserObj
        self.UserNewsTableObj = UserNewsTableObj
        self.UserScrapTableObj = UserScrapTableObj
        
        self.update_status()
        self.make_conf_matrix()
        
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

    def update_status(self) :
        '''
            self.conf_matrix를 새로 update함
        '''
        
        self.newsListTable = pd.read_sql(sql = "select * from news_info_table", con = self.engine)
        self.userNewsTable = pd.read_sql(sql = "select * from user_news_table", con = self.engine)
        self.userScrapTable = pd.read_sql(sql = "select * from user_scrap_table", con = self.engine)
        self.authUserTable = pd.read_sql(sql = 'select username, gender, birthday from auth_user', con = self.engine)

        read_news_length, self.newsList = self.make_ID_dic(userNewsTable, 'news_id')
        existing_user_length, self.userList = self.make_ID_dic(userNewsTable, 'user_id')
        
        genderList = [authUserTable.loc[authUserTable['id'] == userID]['gender'].values[0] for userID in list(self.userList.keys())]
        birthdayList = [authUserTable.loc[authUserTable['id'] == userID]['birthday'].values[0] for userID in list(self.userList.keys())] 
        
        ageGroup_dict = ageGroup_calculate(self.userList, birthdayList)
        gender_dict = {username : gender for username, gender in zip(self.userList, genderList)}

        # confidence = 1.0 + alpha (>= 1.0)
        self.conf_matrix = np.ones(shape = (exsiting_user_length, read_news_length), dtype = np.float32)
    
    def make_conf_matrix(self) :
        self.supplement_conf_matrix_global()
        self.supplement_conf_matrix_local()
    
    def supplement_conf_matrix_global(self) :
        # 뉴스를 성별로 분류
        '''
            1. 어느 뉴스를 읽은 유저들을 모두 불러옴
            2. 그 유저들의 성별을 파악한 후, 비율을 계산함
            3. conf_matrix에 해당 뉴스 열에 성별에 맞게 confidence를 더함
        '''
        ratio_gender = {}
        for newsID in list(self.newsList.keys()) :
            userID_list = self.userNewsTable.loc[self.userNewsTable['news_id'] == newsID]['user_id'].tolist()

            genderGroup = [0, ] * 2
            for userID in userID_list :
                if gender_dict[userID] == '남자' :
                    genderGroup[0] += 1
                elif gender_dict[userID] == '여자' :
                    genderGroup[1] += 1

                try :
                    ratio_gender[newsID] = [round(0.5 * (genderGroup[idx] / sum(genderGroup)), 2) for idx in range(len(genderGroup))]
                    
                except ZeroDivisionError as e :
                    print("ZeroDivisionError")
            
        
        for newsID in list(self.newsList.keys()) :
            # 한 뉴스에 대해서 모든 user들의 가중치 업데이트
            for userID in list(userList.keys()) :
                if (gender_dict[userID] == '남자') :
                    self.conf_matrix[self.userList[userID]][self.newsList[newsID]] += ratio_gender[newsID][0]
                else :
                    self.conf_matrix[self.userList[userID]][self.newsList[newsID]] += ratio_gender[newsID][1]   
                    
        # 뉴스를 연령대로 분류
        '''
            1. 어느 뉴스를 읽은 유저들을 모두 불러옴
            2. 그 유저들의 연령대를 파악한 후, 비율을 계산함
            3. conf_matrix에 해당 뉴스 열에 연령대에 맞게 confidence를 더함
        '''
        ratio_ageGroup = {}
        for newsID in list(newsList.keys()) :
            userID_list = userNewsTable.loc[userNewsTable['news_id'] == newsID]['user_id'].tolist()

            # 10대 이하, 20대, 30대, 40대, 50대, 60대 이상
            ageGroup = [0. ] * 6
            for userID in userID_list :
                ageGroup[ageGroup_dict[userID] - 1] += 1

                ratio_ageGroup[newsID] = [round(0.5 * (ageGroup[idx] / sum(ageGroup)), 2) for idx in range(len(ageGroup))]

        for newsID in list(self.newsList.keys()) :
            # 한 뉴스에 대해서 모든 user들의 가중치 업데이트
            for userID in list(userList.keys()) :
                self.conf_matrix[self.userList[userID]][self.newsList[newsID]] += ratio_ageGroup[newsID][ageGroup_dict[userID] - 1]
           
    def supplement_conf_matrix_local(self) :
        # user가 scrap한 뉴스에 가중치를 부여        
        # user가 scrap을 많이 할수록, confidence에 더해지는 값이 작어짐.
        
        # [[user_id, news_id], [user_id, news_id], [user_id, news_id], ... ]
        scrapList = [[self.userList[user_id], self.newsList[news_id]] for user_id, news_id in 
                             zip(self.userScrapTable['user_id'].tolist(), self.userScrapTable['news_id'].tolist())]
                
        for userID in list(self.userList.keys()) :
            scrapNews = [elements[1] for elements in scrapList if elements[0] == self.userList[userID]]
            nums_of_scrap = len(scrapNews)
            nums_of_readNews = len(self.userNewsTable.loc[self.userNewsTable['user_id'] == userID])
            
            # 1개 이상의 뉴스를 읽었고, 1개 이상의 news를 스크랩했을 경우
            if (readNews and nums_of_scrap) :
                value = ((nums_of_readNews - nums_of_scrap) / float(nums_of_readNews))
                for newsID in scrapNews :
                    self.conf_matrix[ self.userList[userID] ][ newsID ] += value