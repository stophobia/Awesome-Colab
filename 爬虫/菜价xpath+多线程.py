
import requests 
from lxml import etree
import csv
from concurrent.futures import ThreadPoolExecutor

f=open("爬虫/菜价2.csv",mode="w",encoding="utf-8")
csvwriter=csv.writer(f)

def download_one_page(url):
    #拿到页面源代码
    resp=requests.get(url)
    #print(resp.text)
    html = etree.HTML(text=resp.text)
    #table=html.xpath("/html/body/div[2]/div[4]/div[1]/table")
    #print(type(table))#<class 'list'>
    table=html.xpath("/html/body/div[2]/div[4]/div[1]/table")[0]
    #print(type(table))#<class 'lxml.etree._Element'>
    #print(table)#[<Element table at 0x7fcff1450608>]
    trs=table.xpath("./tr")[1:]
    #trs=table.xpath("./tr[position()>1")
    #print(trs)#[<Element tr at 0x7f9d6ea14388>, <Element tr at 0x7f9d6ea14448>, <Element tr at 0x7f9d6ea14348>, <Element tr at 0x7f9d6e791788>, <Element tr at 0x7f9d6e791308>, <Element tr at 0x7f9d6e7a71c8>, <Element tr at 0x7f9d6e7a7148>, <Element tr at 0x7f9d6e7a7308>, <Element tr at 0x7f9d6e7a7348>, <Element tr at 0x7f9d6e7a7208>, <Element tr at 0x7f9d6e7a7388>, <Element tr at 0x7f9d6e7a73c8>, <Element tr at 0x7f9d6e7a7408>, <Element tr at 0x7f9d6e7a7448>, <Element tr at 0x7f9d6e7a7488>, <Element tr at 0x7f9d6e7a74c8>, <Element tr at 0x7f9d6e7a7508>, <Element tr at 0x7f9d6e7a7548>, <Element tr at 0x7f9d6e7a7588>, <Element tr at 0x7f9d6e7a75c8>, <Element tr at 0x7f9d6e7a7608>]
    #print(len(trs))#21
    #拿到每个tr
    for tr in trs:
        text=tr.xpath("./td/text()")
        #对数据进行简单的处理：\\  |  /
        text=(item.replace("\\","").replace("/","").replace("|","") for item in text)
        #print(type(text))#<class 'generator'>
        #把数据存放在csv文件中
        csvwriter.writerow(text)
        #print(list(text))
    print(url.split("/")[-1],"提取完毕")

if __name__=="__main__":
     #for i in range(1,16465):#效率及其低下
      #   download_one_page(f"http://xinfadi.com.cn/marketanalysis/0/list/{i}.shtml")
    with ThreadPoolExecutor(50) as thread:
        for i in range(1,201):#200*20=4000
            thread.submit(download_one_page,f"http://xinfadi.com.cn/marketanalysis/0/list/{i}.shtml" )

print("全部提取完毕！！")


