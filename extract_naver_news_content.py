from fake_useragent import UserAgent
from bs4 import BeautifulSoup as BS
from dateutil.parser import parse

import requests as rq
import datetime
import time
import json
import os, glob
import re
from copy import deepcopy

CATEGORIES = ['정치', '경제', '사회', '생활', '세계', '과학']
CATEGORIES_ENGLISH = ['politics', 'economy', 'society', 'living',
                      'world', 'science']

def save_json_contents(dir_path, date, json_contents) :
    if (os.path.exists(dir_path)) :
        pass
    else :
        os.makedirs(dir_path)

    with open(dir_path + "/" + date + ".json", "w", encoding = 'utf-8') as fp :            
        json_val = json.dumps(json_contents, indent = 4,
                              ensure_ascii = False)
        fp.write(json_val)

def string_filter(strings) :
    #print(strings)
    #print("-----------------------------------------------------------------------")
    strings = re.sub('[\.]+', '.', strings)
    strings = re.sub('[^ 가-힣a-zA-Z0-9.][기자|앵커|서울경제 디센터|뉴스데스크]+[^ 가-힣a-zA-Z0-9.]', '', strings)
    hangul = re.compile('[^ 가-힣a-zA-Z0-9.,]+')
    result = hangul.sub('', strings)

    result = result.split('.')
    for i in range(len(result)) :
        result[i] = result[i].strip()
        if (len(result[i]) <= 15) :
            result[i] = ' '

    result = ". ".join(result)
    return result

def main() :
    global CATEGORIES_KOREAN
    global CATEGORIES_ENGLISH
    
    # fake userAgent 생성
    ua = UserAgent()
    fake_google = ua.google
    fake_headers = {"User-Agent" : fake_google}

    path = "./naver_news/crawling_links/"
    if (os.path.exists(path)) :
        # 경로에 존재하는 모든 json 파일들을 읽어옴
        link_files = glob.glob(path + "*.json")
        link_files.reverse()
    else :
        print("[ {} ] ==> 해당 경로가 존재하지 않습니다.".format(path))
        quit()
        
    contents_path = "./naver_news/news_contents"
    if (os.path.exists(contents_path)) :
        print("[ {} ] ==> 해당 경로가 존재합니다.".format(contents_path))
        pass
    else :
        print("[ {} ] ==> 해당 경로가 존재하지 않습니다.".format(contents_path))
        for category in CATEGORIES_ENGLISH :
            os.makedirs("{}./{}".format(contents_path, category))

    while True :
        try :
            base_date = int(input("기준 날짜(연도월일 8글자 ex : 20160425)    : "))
            break
        except ValueError as e :
            print("날짜를 다시 입력해주세요")
            continue

    # 파일명에서 날짜만 추출
    date_files = [file[28 : 45] for file in link_files]
    date_files = [[int(date_range[ : 8]), int(date_range[9 : ])] for
                       date_range in date_files]

    index = -1
    for idx, lists in enumerate(date_files) :
        if (lists[0] <= base_date) and (base_date <= lists[1]) :
            index = idx
            break

    if (index == -1) :
        print("해당 날짜를 찾을 수 없습니다.")
        quit()
        
    link_files = link_files[index : ]
    base_date = parse(str(base_date))
    
    i = 0
    contents = {}
    for file in link_files :
        with open(file, "r", encoding = 'utf-8') as fp :
            data = fp.read()
            json_data = json.loads(data)

            for day in range(7) :
                temp_time = base_date - datetime.timedelta(days = i)
                date_str = temp_time.strftime('%Y%m%d')
                print("\n                                {}".format(date_str))

                # 파일에 해당 날짜가 존재하면
                if date_str in list(json_data.keys()) :                   
                    date_links = json_data[date_str]

                    # 정치, 경제, 사회, 생활, 세계, 과학
                    for index in range(6) :
                        print("\n", CATEGORIES[index])
                        category_links = date_links[CATEGORIES[index]]
                        dir_path = contents_path + "/{}".format(CATEGORIES_ENGLISH[index])

                        temp = {}
                        for idx, news_info in enumerate(category_links) :
                            title, link = news_info[0], news_info[1]
                            temp['title'] = title
                            temp['link'] = link

                            time.sleep(2)
                            news_res = rq.get(link, headers = fake_headers)
                            soup = BS(news_res.content, 'lxml')
                            [s.extract() for s in soup(['script', 'span', 'a', 'h4'])]

                            try : 
                                texts = soup.find(id = 'articleBodyContents')
                                
                                article = texts.get_text().strip()
                                result = string_filter(article)
                                temp['original'] = article
                                temp['changed'] = result
                                
                                contents[idx] = deepcopy(temp)
                                
                            except AttributeError as e :
                                try :
                                    texts = soup.find(id = 'articeBody') # 연애기사
                                    article = texts.get_text().strip()
                                    result = string_filter(article)

                                    temp['original'] = article
                                    temp['changed'] = result
                                    
                                    contents[idx] = deepcopy(temp)
                                    
                                except AttributeError as e :
                                    temp['original'] = u"본문을 찾을 수 없습니다."
                                    temp['changed'] = u'본문을 찾을 수 없습니다.'
                                    contents[idx] = deepcopy(temp)
                                                           
                        save_json_contents(dir_path, date_str, contents)
                        contents.clear()
                        
                    i += 1
                    
                else :
                    break
            
        print("time.sleep(5)")
        time.sleep(5)
            
if __name__ == "__main__" :
    main()
