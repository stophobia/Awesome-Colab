#协程：单线程条件下，当程序遇到I/O操作时，可以选择切换到其它任务上，微观上串行，切换条件一般是IO操作，宏观上多任务异步并行

import asyncio
import time

async def func1():
    print("Hello , My name is 习包子")
    #time .sleep(3)#当程序出现同步操作时，异步就中断了,9s
    await asyncio.sleep(3)#异步操作的代码：挂起3秒,4.002668380737305s
    print("Hello , My name is 习包子")

async def func2():
    print("Hello , My name is 江蛤蟆")
    #time .sleep(2)
    await asyncio.sleep(2)#异步操作的代码：挂起3秒
    print("Hello , My name is 江蛤蟆")

async def func3():
    print("Hello , My name is 毛腊肉")
    #time .sleep(4)
    await asyncio.sleep(4)#异步操作的代码：挂起3秒
    print("Hello , My name is 毛腊肉")

async def main():
    #第一种写法
    #f1=func1()
    #await f1#一般await挂起操作放在协程对象前面
    #第二种写法（推荐）
    tasks=[
        func1(),#python3.8有警告python3.11後报错
        func2(),
        func3()                        
        #asyncio.create_task(func1()),#python3.8要加上asyncio.create_task()
        #asyncio.create_task(func2()),
        #asyncio.create_task(func3())
    ]
    await asyncio.wait(tasks)
    #task1 = asyncio.create_task(
     #   say_after(1, 'hello'))
 
    #task2 = asyncio.create_task(
     #   say_after(2, 'python'))

if __name__=="__main__":
    t1=time.time()
    print(f"started at {time.strftime('%X')}")
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(main())
    t2=time.time()
    print(f"finished at {time.strftime('%X')}")
    print(t2-t1)

    #g=func()#此时函数是一个异步协程函数，函数执行得到的是一个协程对象
    #print(g)#<coroutine object func at 0x7f7b921aae08>sys:1: RuntimeWarning: coroutine 'func' was never awaited
    #asyncio.run(g)#Python3.7+
    #asyncio.get_event_loop().run_until_complete(g)#python3.6.9#协程程序需要运行需要asyncio模块的支持
    """
    f1=func1()
    f2=func2()
    f3=func3()
    tasks=[f1,f2,f3]
    t1=time.time()
    print(f"started at {time.strftime('%X')}")
 
   
    print(f"finished at {time.strftime('%X')}")
    #一次性启动多个任务（协程）
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(asyncio.wait(tasks))
    t2=time.time()
    print(f"finished at {time.strftime('%X')}")
    print(t2-t1)
    """











