from collections import Counter
from sqlalchemy import create_engine
import pandas as pd

def condition(db_address) :
    engine = create_engine(db_address)
    num_news = len(pd.read_sql("select news_id from news_info_table", con = engine))
    user_id_set = set(pd.read_sql("select user_id from user_news_table", con = engine)['user_id'])
    
    user_read_dic = {}
    unique_newsID = []

    for user_id in user_id_set :
        temp = pd.read_sql("select news_id from user_news_table where user_id = {}".format(user_id), con = engine)['news_id'].values
        count_sum = sum(list(pd.read_sql("select count from user_news_table where user_id = {}".format(user_id), con = engine)['count'].values))
        
        unique_newsID.extend(list(temp))
        user_read_dic[user_id] = temp

    unique_newsID = list(set(unique_newsID))
    print("user_id_set:{}".format(user_id_set))
    
    if (len(unique_newsID) / float(num_news)) < 0.001 :
        print("len(unique_newsID) / float(num_news)) < 0.001 ---> ", len(unique_newsID) / float(num_news))
        return False
    
    total_data = []
    for user_id in list(user_read_dic.keys()) :
        total_data.extend(user_read_dic[user_id])
        
    counts = Counter(total_data)
    one_elements = len([element for element, count in counts.items() if count == 1])    
    common_per = (len(counts) - one_elements) / float(len(counts))
    if (len(counts) == one_elements or one_elements == 0) : # user들 간에 같이 읽은 news가 1개도 없거나 모두 동일 하다면
        print("공통된 원소가 없음")
        return False

    return True