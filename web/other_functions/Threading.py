import threading
from queue import Queue

class ThreadMF(threading.Thread) :
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
                print("Error in ThreadMF")
                print(e)