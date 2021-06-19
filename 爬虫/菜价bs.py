#!usr/bin/python3

from bs4 import BeautifulSoup
import requests
import csv

url = "http://xinfadi.com.cn/marketanalysis/0/list/1.shtml"
response = requests.get(url=url)
f=open("菜价.csv", mode="w") 
csvwriter=csv.writer(f)


page_content = BeautifulSoup(response.text,"html.parser")

#从bs对象中查找数据
#table = page_content.find("table", class_="hq_table")#class是关键字
table = page_content.find("table", attrs={"class":"hq_table"})#和上一行是一个意思，此时可以避免class

trs = table.find_all("tr")[1:]

for tr in trs:#每一行
    tds=tr.find_all("td")#拿到每行中的所有td
    name=tds[0].text#.text拿到被标签标记的内容
    lowest_price=tds[1].text
    highest_price=tds[2].text
    avg_price=tds[3].text
    volume=tds[4].text
    kind=tds[5].text
    date=tds[6].text
    csvwriter.writerow([name,lowest_price,highest_price,avg_price,volume,kind,date])

f.close()
print("Over!")