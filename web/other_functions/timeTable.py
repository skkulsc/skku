import pandas as pd
from polls.models import UserNewsTimeTable

class TimeTable() :
    def __init__(self, engine) :
        self.engine = engine
        self.update_userID()
        self.userReadingDic = dict()
        
    def update_table(self) :
        self.authUserTable = pd.read_sql(sql = 'select id, username from auth_user', con = self.engine) 
        
    def update_userID(self) :
        self.update_table()        
        self.userID_dict = {user_id : username for user_id, username in  zip(self.authUserTable['id'].tolist(), 
                                                                             self.authUserTable['username'].tolist())}
    
    def update_userReadingDic(self, userID, newsID, dateTime) :
        key = (userID, newsID)
        
        if key in list(self.userReadingDic.keys()) : # 이미 존재하는 key라면 reading_time을 table에 추가
            self.record_readingTime(userID, newsID, dateTime)
        else : # 존재하지 않는 key라면 start_time으로 dic에 추가
            self.userReadingDic[(userID, newsID)] = dateTime
        
    def record_readingTime(self, userID, newsID, endTime) :
        startTime = self.userReadingDic[(userID, newsID)]

        # 초 단위로 저장
        data = UserNewsTimeTable(user_id = userID,
                                 news_id = newsID,
                                 reading_time = (endTime - startTime).total_seconds())
        data.save()
        
        
