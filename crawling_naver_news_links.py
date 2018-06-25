from fake_useragent import UserAgent
from bs4 import BeautifulSoup as BS
from dateutil.parser import parse

import requests as rq
import datetime
import time
import json
import os
import copy

# json파일을 저장하는 함수
def save_jsonFile(dir_name, from_, to_, json_content) :
    file_name = dir_name + "/{}_{}.json".format(from_, to_)
    with open(file_name, "w", encoding = "utf-8") as fp :
        json_val = json.dumps(json_content, indent = 4,
                              ensure_ascii = False)
        fp.write(json_val)
    print("                    [ {} ]부터 [ {} ]까지의 뉴스 저장 완료".format(from_, to_))
    return

# 원하는 경로에 폴더를 생성하는 함수
def make_directory(dir_name) :
    # 폴더가 이미 존재하는지 확인(존재하면 is_exist = True)
    is_exist = os.path.exists(dir_name)

    if is_exist :
        print(dir_name, " 폴더가 이미 존재합니다.")
    else :
        print(dir_name, " 폴더가 존재하지 않습니다.")
        print("폴더를 생성합니다.")
        os.makedirs("./{}".format(dir_name))
        print("폴더를 성공적으로 생성하였습니다.")

# 네이버 로봇, 메인, 뉴스, 랭킹 홈페이지 소스를 저장하는 함수
def init_naver(dir_name, fake_headers) :
    url = "https://www.naver.com"
    news_url = "http://news.naver.com"
    robot_url = url + "/robots.txt"

    # robots.txt 사이트
    robot_res = rq.get(robot_url, headers = fake_headers)
    with open(dir_name + "/robots.txt", "w", encoding = "utf-8") as filePointer :
        filePointer.write(robot_res.content.decode('utf-8'))

    # naver 메인 홈페이지
    time.sleep(1)
    main_res = rq.get(url, headers = fake_headers)
    soup = BS(main_res.content, 'lxml')

    with open(dir_name + "/naver_main.txt", "w", encoding = "utf-8") as filePointer :
        filePointer.write(soup.prettify())

    # naver 뉴스 홈페이지
    time.sleep(1)
    news_res = rq.get(news_url, headers = fake_headers)
    soup = BS(news_res.content, 'lxml')

    with open(dir_name + "/naver_news.txt", "w", encoding = 'utf-8') as \
         filePointer :
        filePointer.write(soup.prettify())

    # naver 랭킹 뉴스 홈페이지
    time.sleep(1)
    ranking_news_res = rq.get('http://news.naver.com/main/ranking/popularDay.nhn?rankingType=popular_day',
                              headers = fake_headers)
    soup = BS(ranking_news_res.content, 'lxml')

    with open(dir_name + "/naver_news_ranking.txt", 'w', encoding = 'utf-8') as filePointer :
        filePointer.write(soup.prettify())

def main() :
    # URL의 sectionId
    # ex) 정치 sectionId : 100
    # http://news.naver.com/main/ranking/popularDay.nhn?rankingType=popular_day&sectionId=100&date=20180311
    url_id = {"정치" : 100, "경제" : 101, "사회" : 102, "생활" : 103, "세계" : 104, "과학" : 105}

    dir_name = "./naver_news"
    make_directory(dir_name)

    # fake userAgent 생성
    ua = UserAgent()
    fake_headers = {"User-Agent" : ua.google}
    
    # init_naver(dir_name, fake_headers)

    news_url = "https://news.naver.com"
    ranking_url = news_url + "/main/ranking/popularDay.nhn?rankingType=popular_day"

    # 랭킹뉴스 접근
    ranking_news_res = rq.get(ranking_url,
                              headers = fake_headers)
    soup = BS(ranking_news_res.content, 'lxml')

    start_time = parse(input("크롤링 시작 날짜(연도월일 8글자, ex : 20160525)   : "))
    end_time = parse(input("크롤링 마지막 날짜(연도월일 8글자, ex : 20160523)   : "))       
    delta = start_time - end_time

    # 날짜들의 링크를 저장할 dictionary {'날짜' : '링크'}
    date_links = {}
    for i in range(0, delta.days + 1) :
        temp_time = start_time - datetime.timedelta(days = i)
        date_str = temp_time.strftime('%Y%m%d')
        
        # 해당 날짜의 카테고리 8개를 모두 저장
        temp_links = {}
        for key, value in url_id.items() :
            url = ranking_url + '&sectionId=' + str(value) + '&date=' + date_str
            temp_links[key] = url

        date_links[date_str] = temp_links
    
    # links들을 저장할 폴더를 생성
    crawling_dir = dir_name + "/crawling_links"
    make_directory(crawling_dir)
    
    temp = start_time   
    saving_links = {} # 7일 단위의 각 카테고리들의 뉴스 링크(30개씩)들을 저장
    category_links = {}
    
    try :
        for date, links in date_links.items() :
            print("크롤링 날짜 => ", date)
            # 특성 시점에 대한 모든 기사(30개)들을 저장
            for category, category_link in links.items() :
                time.sleep(2)
                category_res = rq.get(category_link, headers = fake_headers)
                
                soup = BS(category_res.content, 'lxml')

                temp_links = soup.select('#wrap table td.content > div.content ol.ranking_list > li > div.ranking_text > div.ranking_headline > a')
                temp_links = [[link.attrs['title'], news_url + link.attrs['href']] for link in temp_links]

                # {"정치" : [30개의 링크 리스트], "경제" : [30개의 링크 리스트], ...}
                category_links[category] = temp_links

            # 특정 시점에 대한 모든 기사들을 다 저장했다면, 또 다른 dict에 다른 일정의 기사들을 저장
            # {"20180312" : {"정치" : [...], "경제" : [...], ...}, "20180311" : {"정치" : [...], ...}, ...}
            # dictionary는 mutable 타입
            saving_links[date] = copy.deepcopy(category_links)
            category_links.clear()

            # 마지막 날짜까지 저장했음
            if (parse(date) == end_time) :
                save_jsonFile(crawling_dir, parse(date).strftime('%Y%m%d'), temp.strftime('%Y%m%d'), saving_links)
                
            # 7일 단위로 저장
            elif ((temp - parse(date)).days + 1 == 7) :
                save_jsonFile(crawling_dir, parse(date).strftime('%Y%m%d'), temp.strftime('%Y%m%d'), saving_links)

                # saving_links에 저장된 모든 요소들을 제거
                saving_links.clear()
                temp = temp - datetime.timedelta(days = 7)
                
    except KeyboardInterrupt as e:
        print("KeyboardInterrupt")

if __name__ == "__main__" :
    main()
    
