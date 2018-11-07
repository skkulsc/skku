import os, glob
import json
import random
import operator
import numpy as np

CATEGORIES = ['politics', 'economy', 'society', 'living', 'world', 'science']
LABEL = {'politics' : 0, 'economy' : 1, 'society' : 2,
         'living' : 3, 'world' : 4, 'science' : 5}

freq_dic = {}                       # 단어의 빈도 수를 저장하는 딕셔너리
word_dic = {"PAD_" : 0, "UNK_" : 1} # {id : 단어}
word_dic_reverse = {}               # {단어 : id}
ID = 2                              # (0 : PAD_, 1 : UNK_)
NUM_WORDS = 10                      # 최소 NUM_WORDS 이상 나온 단어들만 사전에 저장

content_path = "./naver_news/preprocessing_contents/"
training_path = "./naver_news/training/words_" + str(NUM_WORDS)
dic_file = training_path + "/word-dic.json"
dic_reverse_file = training_path + "/word-dic-reverse.json"
data_file = training_path + "/data/"

def save_json_contents(dir_path, file_name, json_contents) :
    if (os.path.exists(dir_path)) :
        pass
    else :
        os.makedirs(dir_path)

    with open(dir_path + "/" + file_name + ".json", "w", encoding = 'utf-8') as fp :            
        json_val = json.dumps(json_contents, indent = 4,
                              ensure_ascii = False)
        fp.write(json_val)

def make_freq_dic(texts) :
    global freq_dic
    global ID

    if (len(texts) <= 0) :
        return -1
    
    texts = texts.strip()
    words = texts.split(" ")
    
    for n in words :
        n = n.strip()
        # n이 공백문자라면
        if n == '' :
            continue

        # n이 단어 사전에 없다면
        if not n in freq_dic :
            freq_dic[n] = 1

        # 이미 단어 사전에 존재하는 단어라면 1을 증가시킴  
        else :
            freq_dic[n] += 1

def make_word_dic(texts) :
    global word_dic
    global MAX_LENGTH

    if (len(texts) <= 0) :
        return -1

    words = texts.strip().split(" ")

    results = []
    for n in words :
        # 단어 사전에 그 단어가 존재하지 않으면
        if not n in word_dic :
            results.append(word_dic["UNK_"])
        else :
            results.append(word_dic[n])
            
    return results
    
def words_to_number(files) :
    X = [] # input
    Y = [] # label

    random.shuffle(files)
    
    # 단어를 숫자로 변환
    for list_ in files :
        file = list_[0]
        label = list_[1]

        with open(file, "r", encoding = 'utf-8') as fp :
            texts = fp.read()
            json_data = json.loads(texts)

            for key in list(json_data.keys()) :
                results = make_word_dic(json_data[key])

                if (results == -1) :
                    pass
                else :
                    X.append(results)
                    Y.append(label)

    return X, Y

def number_to_words(number_list) :
    global word_dic_reverse
    X = [] # input

    for word in number_list :
        X.append(word_dic_reverse[word])

    return X
    
def main() :
    global NUM_WORDS
    global content_path
    global dic_file
    global word_dic
    global freq_dic
    global ID
    global dic_reverse_file
    global word_dic_reverse
    
    dir_path = [content_path + category  + "/" for category in CATEGORIES]
    
    train_files = []
    test_files = []
    train_test_ratio = 0.8 # 트레이닝 데이터와 테스트 데이터를 나눌 비율 [ 0.8 : 0.2 ]
    for idx, path in enumerate(dir_path) :
        temp = glob.glob(path + "*.json")
        length = len(temp)
        for i in range(length) :
            temp[i] = [temp[i], idx]
            
        random.shuffle(temp)

        # train data와 test 데이터 나누기
        train_test_boundary = int(length * train_test_ratio)    
        test_files.extend(temp[train_test_boundary : ])
        train_files.extend(temp[ : train_test_boundary])
                                         
        print("train_files : ", len(train_files), "    test_files : ", len(test_files))
        
        del temp
    
    # 단어 빈도수 조사
    for list_ in train_files :
        file = list_[0]
        with open(file, "r", encoding = 'utf-8') as fp :
            texts = fp.read()
            json_data = json.loads(texts)

            for key in list(json_data.keys()) :
                make_freq_dic(json_data[key])

    for list_ in test_files :
        file = list_[0]
        with open(file, "r", encoding = 'utf-8') as fp :
            texts = fp.read()
            json_data = json.loads(texts)

            for key in list(json_data.keys()) :
                make_freq_dic(json_data[key])                

    for key in list(freq_dic.keys()) :
        if freq_dic[key] < NUM_WORDS : # 빈도수가 NUM_WORDS 이하인 단어들은 모두 제외시킴
            del freq_dic[key]
    print("NUM_WORDS = {} =====> len(freq_dic) = {}".format(NUM_WORDS, len(freq_dic)))

    # 빈도수로 단어를 정렬
    sorted_dict = sorted(freq_dic.items(), key = operator.itemgetter(1), reverse = True)

    # 빈도수를 기준으로 사전에 순서대로 저장함
    for lists in sorted_dict :
        word_dic[lists[0]] = ID
        ID += 1

    word_dic_reverse = {v : k for k, v in word_dic.items()} # 역순
    print(word_dic_reverse[0], word_dic_reverse[1])

    # 단어를 숫자로 변환
    train_X, train_Y = words_to_number(train_files)
    test_X, test_Y = words_to_number(test_files)
            
    
    if os.path.exists(training_path) :
        pass
    else :
        os.makedirs(training_path)

    if os.path.exists(data_file) :
        pass
    else :
        os.makedirs(data_file)
          
    json.dump(word_dic, open(dic_file, "w", encoding = 'utf-8'))
    del(word_dic)
    print("word_dic 저장")

    json.dump(word_dic_reverse, open(dic_reverse_file, "w", encoding = 'utf-8'))
    print("word_dic_reverse 저장")

    # 역순
    train_reverse_X = []
    test_reverse_X = []
    for X in train_X :
        train_reverse_X.append(number_to_words(X))
    for X in test_X :
        test_reverse_X.append(number_to_words(X))
        
    file_name = data_file + "train_result.json"
    json.dump({"X" : train_X, "Y" : train_Y},
              open(file_name, "w", encoding = 'utf-8'))
    
    file_name = data_file + "test_result.json"
    json.dump({"X" : test_X, "Y" : test_Y},
              open(file_name, "w", encoding = 'utf-8'))

    file_name = data_file + "train_reverse_result.json"
    json.dump({"X" : train_reverse_X, "Y" : train_Y},
              open(file_name, "w", encoding = 'utf-8'))
    
    file_name = data_file + "test_reverse_result.json"
    json.dump({"X" : test_reverse_X, "Y" : test_Y},
              open(file_name, "w", encoding = 'utf-8'))

if __name__ == "__main__" :
    main()
