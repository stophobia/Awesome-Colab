

#第一种

from multiprocessing  import Process
from threading import Thread

def func(name):  #传参
    for i in range(1000):
        print(name, i) 

if __name__=="__main__":
    t1=Thread(target=func, args=("蒋介石",))#传递参数必须是元组，且一个参数必须加，
    t1.start()
    t2=Thread(target=func, args=("孙逸仙",))#传递参数必须是元组，且一个参数必须加，
    t2.start()

    #process=Process(target=func)#创建线程并给线程安排任务
    #process.start()#多线程状态就绪，具体的执行时间由CPU决定
    for i in  range(1000):
        print("主进程",i)


#第二种
"""
class MyProcess(Process):
    def run(self):              #固定的->当线程被执行时被执行的就是run()
        for i in range(1000):
            print("子线程",i)

if __name__=="__main__":
    t=MyProcess()
    #t.run()#方法的调用->单线程
    t.start()#开启线程
    for i in  range(1000):
        print("主线程",i)
"""

