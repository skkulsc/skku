from polls.models import NewsInfoTable, AuthUser, UserNewsTable, UserNewsTimeTable
from other_functions import Recommendation, timeTable, Threading
from sqlalchemy import create_engine

db_adress = 'mysql+pymysql://lee:Skkuastro561!@35.230.61.91/rec_system?charset=utf8mb4'
engine = create_engine(db_adress)

# Rec_system class
try :
    rec_system = Recommendation.Rec_system(NewsInfoTable.objects, AuthUser.objects, UserNewsTable.objects, 
                                       engine)    
    rec_system.do_MF()
    
except Exception as e :
    print("\nError in initial rec_system --- other_functions.start.py")
    print("Error:\n{}\n".format(e))

# TimeTable class
try :
    time_table = timeTable.TimeTable(engine)
    time_table.update_userID()

except Exception as e :
    print("\nError in initial time_table --- other_functions.start.py")
    print("Error:\n{}\n".format(e))
    
# Rec_system을 위한 queue
try :
    # 순차적으로 matrix factorizaton을 수행함
    recSystem_queue = Threading.Queue()    
    recSystem_thread = Threading.ThreadMF(recSystem_queue)
    recSystem_thread.setDaemon(True)
    recSystem_thread.start()
except Exception as e :
    print("\nError in initial recSystem_queue --- other_functions.start.py")
    print("Error:\n{}\n".format(e))

# TimeTable을 위한 queue
try :
    # 순차적으로 update를 수행함
    timeTable_queue = Threading.Queue()
    timeTable_thread = Threading.ThreadMF(timeTable_queue)
    timeTable_thread.setDaemon(True)
    timeTable_thread.start()

except Exception as e :
    print("\nError in initial timeTable_queue --- other_functions.start.py")
    print("Error:\n{}\n".format(e))