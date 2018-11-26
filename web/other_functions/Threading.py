import threading
from queue import Queue

class ThreadMF_forRec(threading.Thread) :
    def __init__(self, queue) :
        threading.Thread.__init__(self)
        self.queue = queue
        
    def run(self) :
        while True :
            try :
                rec_system, userID, iters, rank, lr, lr_decay, reg, step, opt = self.queue.get()
                rec_system.do_CBF(userID)
                rec_system.do_MF()  
                self.queue.task_done()
                
            except Exception as e :
                print("\nError in ThreadMF_forRec")
                print("Error : \n{}\n".format(e))
                
class ThreadMF_forTime(threading.Thread) :
    def __init__(self, queue) :
        threading.Thread.__init__(self)
        self.queue = queue
    
    def run(self) :
        while True :
            try :
                time_table, userID, newsID, Time = self.queue.get()
                time_table.update_userReadingDic(userID, newsID, Time)
                self.queue.task_done()
                
            except Exception as e :
                print("\nError in ThreadMF_forTime")
                print("Error : \n{}\n".format(e))