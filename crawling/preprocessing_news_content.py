from konlpy.tag import Okt

import json
import re
import os, glob
import time
import threading

def preprocessing_contents(category, original_paths, preprocessing_paths, start_time, end_time, function) :
    print(original_paths, " 시작")
    
    if (os.path.exists(preprocessing_paths)) :
        pass
    else :
        print(preprocessing_paths, " 생성")
        os.makedirs(preprocessing_paths)

    files = glob.glob(original_paths + "*.json") # 모든 파일을 불러옴
    files.sort(reverse = True)
    
    # 파일명에서 날짜만 추출
    dates = [int(re.findall('\d+', file)[0]) for file in files]
    dates.sort(reverse = True)

    from_ = -1; to_ = -1;
    for idx, date in enumerate(dates) :
        if (start_time == date) :
            from_ = idx
        if (end_time == date) :
            to_ = idx
            
        if (from_ != -1 and to_ != -1) :
            break

    if (from_ == -1 or to_ == -1) :
        print("해당 날짜를 찾을 수 없습니다.")
        print(original_paths, " 종료")
        return

    files = files[from_ : to_ + 1]           
    for file in files :
        try :
            contents = []
            with open(file, "r", encoding = 'utf-8') as fp :
                data = fp.read()
                json_data = json.loads(data)
                date = int(re.findall('\d+', file)[0])
            
                for idx, key in enumerate(list(json_data.keys())) :
                    original_data = json_data[key]["original"].strip()
                    title = json_data[key]["title"].strip()
                    if (original_data == "본문을 찾을 수 없습니다.") :
                        pass
                    else :
                        title_malist = function.pos(title, norm = True, stem = True)
                        malist = function.pos(original_data, norm = True, stem = True)
                
                        result = []
                        for word in title_malist :
                            if word[1] not in ["Punctuation", "Eomi", "PreEomi",
                                        "Unknown", "KoreanParticle",
                                        "Foreign", "Email", "URL",
                                        "Hashtag", "ScreenName"] :
                                result.append((word[0] + "/" + word[1]).lower())
                    
                        for word in malist :
                            if word[1] not in ["Punctuation", "Eomi", "PreEomi",
                                        "Unknown", "KoreanParticle",
                                        "Foreign", "Email", "URL",
                                        "Hashtag", "ScreenName"] :
                                result.append((word[0] + "/" + word[1]).lower())

                        contents.append(category + "\t" + str(date) + "\t" + title + "\t" + " ".join(result))

                with open(preprocessing_paths + "/" + category + "_" + str(date) + ".txt", "w",
                      encoding = 'utf-8') as inner_fp :
                    inner_fp.write("\n".join(contents))
            
            del contents
        except Exception as e :
            print("\nError 발생!!")
            print("original_path : ", original_path)
            print("file name : ", file)
            print("error:\n{}\n\n".format(e))

    print(original_paths, " 종료")
                       
def main() :
    twitter = Okt()
    CATEGORIES = ['politics', 'economy', 'society', 'living', 'world', 'science']
    
    dir_paths = ["./naver_news/news_contents/" + category + "/"
                 for category in CATEGORIES]
    preprocessing_paths = ["./naver_news/preprocessing_contents/" +
                           category + "/" for category in CATEGORIES]
    
    start_date = []
    end_date = []
    
    for i in range(6) :
        print(CATEGORIES[i], " 기사")
        start, end = (input("언제부터 언제까지 (ex : 20180625 20160412) : ")).split()
        if (start < end) :
            start, end = end, start
            
        start_date.append(start)
        end_date.append(end)
        print()

    for i in range(6) :
        threading.Thread(target = preprocessing_contents,
                         args = (CATEGORIES[i], dir_paths[i], preprocessing_paths[i],
                                 int(start_date[i]), int(end_date[i]), twitter)).start()

if __name__ == "__main__" :
    main()
