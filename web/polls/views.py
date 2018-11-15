from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseRedirect
from django.utils import timezone

from .models import NewsInfoTable, AuthUser, UserNewsTable, UserScrapTable
from other_functions import Recommendation, Threading

import pandas as pd
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
    post_keys = list(request.POST.keys())
    
    if ('scrap' in post_keys) :
        print("request.POST['scrap'] : ", request.POST['scrap'])
        
        return HttpResponse(status=201)
        
    elif ('user_read_news_ID' in post_keys) :
        news_id = int(request.POST['user_read_news_ID'])
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

def scrap(request) :
    what, newsID = request.POST['scrap'].split('; ')
    userID = AuthUser.objects.filter(username = request.user)[0].id
    user_movie_row = UserScrapTable.objects.filter(user = userID, news = newsID)
    
    # 이미 scrap한 적이 있는 뉴스임
    if (len(user_movie_row)) :
        # 이미 scrap한 뉴스를 한 번 더 scrap할려고 할 시, Error 메시지를 보냄
        if (what == 'add') :
            return render(request, 'polls/scrap.html', {'cannot_add' : True})
        
        # UserScrapTable에 추가
        else :
            UserScrapTable.objects.filter(user = AuthUser.objects.get(id = userID),
                                          news = NewsInfoTable.objects.get(news_id = newsID)).delete()
            
            return render(request, 'polls/scrap.html', {'info' : 'delete',
                                                        'username' : request.user,
                                                        'news_id' : newsID,
                                                        'time' : timezone.localtime()
                                                       })

    # scrap한 적이 없는 뉴스임
    else :
        # scrap한 적이 없는 뉴스를 삭제하려고 할 시, Error 메시지를 보냄
        if (what == 'cancle') :
            return render(request, 'polls/scrap.html', {'cannot_cancle' : True})
        
        # UserScrapTable에서 제거
        else :
            # default directory = '기본 폴더'
            data = UserScrapTable(user = AuthUser.objects.get(id = userID),
                                 news = NewsInfoTable.objects.get(news_id = newsID),
                                 scrap_time = timezone.localtime())
            data.save()
            
            return render(request, 'polls/scrap.html', {'info' : 'add',
                                                        'username' : request.user,
                                                        'news_id' : newsID,
                                                        'time' : timezone.localtime()
                                                       })

    print("what = {} - userID = {} - newsID = {}".format(what, userID, newsID))
    
def show_scrapNews(request) :
    return HttpResponse("")
