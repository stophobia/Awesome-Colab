#!usr/bin/python3

import requests
from bs4 import BeautifulSoup
import time

url="https://www.umei.net/bizhitupian/weimeibizhi/"

response=requests.get(url=url)
response.encoding="utf-8"

main_page=BeautifulSoup(response.text,"html.parser")

a_list = main_page.find("div", class_="TypeList").find_all("a")

for a in a_list:
    href="https://www.umei.net"+a.get("href") #直接通过get就可以拿到属性值
    #拿到子页面的源代码
    child_page_response=requests.get(href)
    child_page_response.encoding="utf-8"
    child_page_text=child_page_response.text
    #从子页面中拿到图片的下载地址
    child_page=BeautifulSoup(child_page_text,"html.parser")
    p = child_page.find("p", align="center")
    img=p.find("img")
    src=img.get("src")
    #下载图片
    img_resp=requests.get(src)
    #img_resp.content这里拿到的是字节
    img_name=src.split('/')[-1]
    with open("img/"+img_name, mode="wb") as f:
        f.write(img_resp.content)
    
    print("over!!",img_name)
    time.sleep(1)
   
print("All over!")

