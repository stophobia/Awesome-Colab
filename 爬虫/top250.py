#!usr/bin/python3

import requests
import re
import csv


def request_douban(page):
    url = "https://movie.douban.com/top250?start="+str(25*page)+"&filter="
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.101 Safari/537.36"}
    try:
        response = requests.get(url=url, headers=headers)
        if response.status_code == 200:
           return response.text
    except requests.RequestException:
       return None
   
for page in range(10):
    page_content = request_douban(page)
    obj = re.compile(r' <div class="hd">.*? <span class="title">(?P<name>.*?)</span>.*?<div class="bd">.*?<br>(?P<year>.*?)&nbsp;.*?<div class="star">.*?<span class="rating_num" property="v:average">(?P<rate>.*?)</span>.*?<span>(?P<num>.*?)人评价</span>',re.S)
    f = open("movietop250.csv", "a+")
    csvwriter = csv.writer(f)
    result = obj.finditer(page_content)
    for it in result:
        dic = it.groupdict()
        dic['year'] = dic['year'].strip()
        csvwriter.writerow(dic.values())


#response = requests.get(url=url, headers=headers)



#page_content = resp.text

#obj = re.compile(r' <li>.*?<div class="item">.*?<span class="title">(?P<name>.*?)</span>',re.S)
#obj = re.compile(r' <div class="hd">.*? <span class="title">(?P<name>.*?)</span>.*?<div class="bd">.*?<br>(?P<year>.*?)&nbsp;.*?<div class="star">.*?<span class="rating_num" property="v:average">(?P<rate>.*?)</span>.*?<span>(?P<num>.*?)人评价</span>',re.S)


#f = open("movietop250.csv", "w")
#csvwriter = csv.writer(f)

#result = obj.finditer(page_content)
#for it in result:
 #   print(it.group('name'))
  #  print(it.group('year').strip())
   # print(it.group('num'))
    #print(it.group('rate'))
#for it in result:
 #   dic = it.groupdict()
  #  dic['year'] = dic['year'].strip()
   # csvwriter.writerow(dic.values())

f.close()

print('Over!')