from konlpy.tag import Okt
from concurrent.futures import ProcessPoolExecutor

import concurrent
import multiprocessing
import json
import re
import os, glob

def preprocessing_contents(category, original_paths, preprocessing_paths, start_date, end_date) :
    function = Okt()
    if (os.path.exists(preprocessing_paths)) :
        pass
    else :
        os.makedirs(preprocessing_paths)

    files = glob.glob(original_paths + "*.json") # 모든 파일을 불러옴
    files.sort(reverse = True)
    
    # 파일명에서 날짜만 추출
    dates = [int(re.findall('\d+', file)[0]) for file in files]
    dates.sort(reverse = True)

    from_ = -1; to_ = -1
    for idx, date in enumerate(dates) :
        if (start_date == date) :
            from_ = idx
        if (end_date == date) :
            to_ = idx
            
        if (from_ != -1 and to_ != -1) :
            break

    if (from_ == -1 or to_ == -1) :
        if from_ == -1 :
            print("[ {} ] - 해당 날짜를 찾을 수 없습니다.".format(start_date))
        else :
            print("[ {} ] - 해당 날짜를 찾을 수 없습니다.".format(end_date))
        return "{} 종료".format(original_paths)

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
                    original_data = original_data.replace('\n', ' ')
                    title = json_data[key]["title"].strip()
                    if (original_data == "본문을 찾을 수 없습니다.") :
                        pass
                    else :
                        title_malist = function.pos(title, norm = True, stem = True)
                        malist = function.pos(original_data, norm = True, stem = True)

                        result = [(word[0].strip() + "/" + word[1]).lower() for word in title_malist if word[0].strip() != '']
                        result.extend([(word[0].strip() + "/" + word[1]).lower() for word in malist if word[0].strip() != ''])

                        contents.append((category + "\t" + str(date) + "\t" + title + "\t" + " ".join(result)).strip())

                with open(preprocessing_paths + "/" + category + "_" + str(date) + ".txt", "w", encoding = 'utf-8') as inner_fp :
                    inner_fp.write("\n".join(contents))
            del contents

        except Exception as e :
            print("original_path : ", original_paths)
            print("file name : ", file)
            print("error:\n{}\n\n".format(e))

    return "{} 종료".format(original_paths)
           
def main() :
    CATEGORIES = ['politics', 'economy', 'society', 'living', 'world', 'science']   
    dir_paths = ["./naver_news/news_contents/" + category + "/"
                 for category in CATEGORIES]
    preprocessing_paths = ["./naver_news/preprocessing_contents/" + category + "/" for category in CATEGORIES]
    
    start_date = []
    end_date = []
    
    for i in range(6) :
        print(CATEGORIES[i], " 기사", end = ' ')
        start, end = (input("언제부터 언제까지 (ex : 20180625 20160412) : ")).split()
        if (start < end) :
            start, end = end, start
            
        start_date.append(start)
        end_date.append(end)
        print()

    num_threads = min(multiprocessing.cpu_count(), len(CATEGORIES))
    print("num_threads : ", num_threads)

    with ProcessPoolExecutor(max_workers = num_threads) as executor :
        future_to_index = {executor.submit(preprocessing_contents, 
                            CATEGORIES[i], dir_paths[i], preprocessing_paths[i], int(start_date[i]), int(end_date[i])) : i
                            for i in range(num_threads)}
        for future in concurrent.futures.as_completed(future_to_index) :
            index = future_to_index[future]
            try :
                print(future.result())
            except Exception as e :
                print('%r generated an exception: %s'%(index, e))

if __name__ == "__main__" :
    main()
