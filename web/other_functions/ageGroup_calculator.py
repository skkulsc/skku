'''
    https://stackoverflow.com/questions/22344244/age-calculator-in-python-from-date-mm-dd-yyyy-and-print-age-in-years-only
'''

from datetime import datetime

def ageGroup_calculate(usernameList, birthList) :
    ageGroup_dict = dict()
    current_time = datetime.date(datetime.now())
    
    for username, birthDate in zip(usernameList, birthList) :
        raw_age = (current_time - birthDate).days / 365
        ageGroup = int(raw_age / 10)
        if (ageGroup < 1) : # 10대 이하
            ageGroup_dict[username] = 1

        elif (ageGroup > 6) : # 60대 이상
            ageGroup_dict[username] = 6
            
        else :
            ageGroup_dict[username] = ageGroup
            
    return ageGroup_dict