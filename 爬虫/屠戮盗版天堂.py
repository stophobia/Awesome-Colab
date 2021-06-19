#!usr/bin/python3

import  requests
import re

domain = "https://dytt89.com/"
headers = {
    "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.101 Safari/537.36"
}

response = requests.get(url=domain, headers=headers, verify=False)#去掉安全验证

response.encoding = 'gb2312'

page_content = response.text
#拿到ul里的li
obj1 = re.compile(r'2021必看热片.*?<ul>(?P<ul>.*?)<ul>', re.S)
obj2 = re.compile(r"<a href='(?P<href>.*?)'.*?</a>", re.S)
obj3 = re.compile(r'◎片　　名(?P<movie>.*?)<br />.*?<td style="WORD-WRAP: break-word" bgcolor="#fdfddf"><a href="(?P<download>.*?)">',re.S)



result1 = obj1.finditer(page_content)
child_href_list=[]
for it in result1:
    ul=it.group('ul')
    #提取子页面连接
    result2 = obj2.finditer(ul)
    for it2 in result2:
        #拼接子页面的url地址
        child_href=domain+ it2.group("href").strip('/')
        child_href_list.append(child_href)#把子页面链接保存起来
        #print("https://dytt89.com"+str(it2.group('href')))


#提取子页面内容

for href in child_href_list:
    child_response=requests.get(href, verify=False)
    child_response.encoding='gb2312'
    result3=obj3.search(child_response.text)
    print(result3.group('movie'))
    print(result3.group('download'))
    #break




