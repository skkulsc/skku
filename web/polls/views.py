from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseRedirect
from django.utils import timezone

from .models import UserScrapTable
from other_functions.start import *

import threading
import pandas as pd
from datetime import datetime
   
def index(request, con = 0) :        
    global engine
    global time_table
    global timeTable_queue
    
    '''
        user가 로그인에 성공한 뒤 호출되는 함수
    '''
    if 'back' in list(request.POST.keys()) : # 어떤 news를 읽은 뒤 다시 /aritlces 로 이동하는 경우 읽은 시간을 체크
        userID = AuthUser.objects.filter(username = request.user)[0].id
        newsID = int(request.POST['back'])
        timeTable_queue.put([time_table, userID, newsID, datetime.now()])
        
    print("--------------------> [{}] user가 접속함 ---- {}".format(
        request.user, datetime.now().strftime("%Y/%m/%d_%H:%M:%S")))
                
    userID = AuthUser.objects.filter(username = request.user)[0].id

    
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
    global recSystem_queue
    global time_table
    global timeTable_queue
    
    userID = AuthUser.objects.filter(username = request.user)[0].id
    post_keys = list(request.POST.keys())
    
    read_data = dict()
    
    if 'scrap' in post_keys :
        news_id = int(request.POST['scrap'])
    elif 'user_read_news_ID' in post_keys :
        news_id = int(request.POST['user_read_news_ID'])
        
    timeTable_queue.put([time_table, userID, news_id, datetime.now()])

    news_data = NewsInfoTable.objects.filter(news_id = news_id).values()

    print("--------------------> [{}] user가 [{}] id의 news를 읽음 --- {}".format(
        request.user, news_id, datetime.now().strftime("%Y/%m/%d_%H:%M:%S")))
    
    if ('user_read_news_ID' in post_keys) :
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
            recSystem_queue.put([rec_system, userID, 200, 100, 0.001, 0.9, 0.01, 10, 'adam'])

        except Exception as e :
            print("\nError in show_content")
            print(e, "\n")

    read_data['read_data'] = news_data
    read_data['newsID'] = news_id
    if ('scrap' in post_keys) :
        read_data['scrap'] = True
    else :
        read_data['scrap'] = False
        
    return render(request, 'polls/content.html', read_data)

def scrap(request) :
    post_list = request.POST['scrap'].split('; ')
    
    what = post_list[0]
    newsID = post_list[1]
        
    userID = AuthUser.objects.filter(username = request.user)[0].id
    user_movie_row = UserScrapTable.objects.filter(user = userID, news = newsID)
    
    # 이미 scrap한 적이 있는 뉴스임
    if (len(user_movie_row)) :
        # 이미 scrap한 뉴스를 한 번 더 scrap할려고 할 시, Error 메시지를 보냄
        if (what == 'add') :
            return render(request, 'polls/scrap.html', {'cannot_add' : True})
        
        # UserScrapTable에서 삭제
        else :
            UserScrapTable.objects.filter(user = AuthUser.objects.get(id = userID),
                                          news = NewsInfoTable.objects.get(news_id = newsID)).delete()
            
            if (len(post_list) == 2) :
                return render(request, 'polls/scrap.html', {'info' : 'delete',
                                                            'username' : request.user,
                                                            'news_id' : newsID,
                                                            'time' : timezone.localtime()
                                                           })
            elif (len(post_list) == 3) : # scrapNews 페이지에서 삭제를 한 경우
                return show_scrapNews(request)
            
            else :
                return HttpResponse("Error in post_list")

    # scrap한 적이 없는 뉴스임
    else :
        # scrap한 적이 없는 뉴스를 삭제하려고 할 시, Error 메시지를 보냄
        if (what == 'cancle') :
            return render(request, 'polls/scrap.html', {'cannot_cancle' : True})
        
        # UserScrapTable에 추가
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
    
def show_scrapNews(request) :
    global engine
    
    userID = AuthUser.objects.filter(username = request.user)[0].id

    sql = "select * from user_scrap_table where user_id = {} order by scrap_time DESC".format(userID)      
    user_scrap_news = pd.read_sql(sql, con = engine)
    scrapNews = dict()
    scrapNews['scrapNews'] = []
    for idx, row in user_scrap_news.iterrows() :
        scrapNews['scrapNews'].append(NewsInfoTable.objects.get(news_id = row[2]))
        
    print("scrapNews:\n{}\n".format(scrapNews))
    
    return render(request, 'polls/scrapNews.html', scrapNews)
