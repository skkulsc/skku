from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.utils import timezone

from .models import NewsInfoTable, AuthUser, UserNewsTable
from other_functions import Recommendation, Threading

import pandas as pd
import re
import datetime
import threading
from sqlalchemy import create_engine

try :
    db_adress = '...'
    rec_system = Recommendation.Rec_system(NewsInfoTable.objects, AuthUser.objects, UserNewsTable.objects, 
                                       create_engine(db_adress))
    
    rec_system.do_MF()
except Exception as e :
    print("\nError in initial rec_system --- views.py")
    print("Error:\n{}\n".format(e))
    
try :
    # 순차적으로 matrix factorizaton을 수행함
    queue = Threading.Queue()    
    t = Threading.ThreadMF(queue)
    t.setDaemon(True)
    t.start()
except Exception as e :
    print("\nError in initial Queue --- views.py")
    print("Error:\n{}\n".format(e))

def index(request, con = 0) :
    
    global db_address
    '''
        user가 로그인에 성공한 뒤 호출되는 함수
    '''
    
    print("--------------------> [{}] user가 접속함 ---- {}".format(
        request.user, datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S")))
                
    userID = AuthUser.objects.filter(username = request.user)[0].id
    engine = create_engine(db_adress)
    
    sql = "select * from user_news_table where user_id = {} order by read_time DESC".format(userID)  
    user_read_news = pd.read_sql(sql, con = engine)
    
    newsId_list = list(user_read_news['news_id'])

    context = {}
    t_context = threading.Thread(target = rec_system.recommend, 
                                 args = (userID, newsId_list, context))
    
    t_context.start()
    t_context.join()
    
    if (con == 0 ) :
        return render(request, 'polls/index.html', context)
    else :
        return context

def show_content(request) :
    
    '''
        user가 어떤 news를 클릭했을 때, 호출되는 함수
    '''
    global rec_system
    global queue
    
    userID = AuthUser.objects.filter(username = request.user)[0].id
    news_id = int(re.findall('\d+', request.body.decode('utf-8'))[-1])
    news_data = NewsInfoTable.objects.filter(news_id = news_id).values()

    print("--------------------> [{}] user가 [{}] id의 news를 읽음 --- {}".format(
        request.user, news_id, datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S")))
    try :
        user_movie_row = UserNewsTable.objects.filter(user = userID, news = news_id)
        length = len(user_movie_row)
        
        # insert data
        if (length == 0) :
            data = UserNewsTable(user = AuthUser.objects.get(id = userID),
                                 news = NewsInfoTable.objects.get(news_id = news_id),
                                 count = 1,
                                read_time = timezone.localtime())
            data.save()   
        else :
            data = user_movie_row[0]
            if (data.count >= 5) : # 읽은 횟수는 최대 5번까지만 기록함
                data.count = 5
                data.read_time = timezone.localtime()
            else :
                data.count += 1
                data.read_time = timezone.localtime()
                
            data.save()
    
        rec_system.update_table(NewsInfoTableObj = None, AuthUserObj = None, UserNewsTableObj = UserNewsTable.objects)
        queue.put([rec_system, userID, 200, 100, 0.001, 0.9, 0.01, 10, 'adam'])

    except Exception as e :
        print("\nError in show_content")
        print(e, "\n")

    read_data = {'read_data' : news_data}
    
    return render(request, 'polls/content.html', read_data)
