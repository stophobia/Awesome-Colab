#!/usr/bin/python3

import requests

query = input("Please input a word: ")

payload = {"kw" :  query}

url =  "https://fanyi.baidu.com/sug"

headers = {
   "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.101 Safari/537.36"
}

# 发送post请求，发送的数据必须放在字典里，通过data参数进行传递

resp = requests.post(url=url, data=payload)

json = resp.json()


print(json)



