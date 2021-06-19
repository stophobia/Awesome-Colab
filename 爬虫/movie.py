#!usr/bin/python3

import requests 

url = "https://movie.douban.com/j/new_search_subjects"

# 重新封装参数

params = {
    "sort" :  "T",
     "range": "0,10",
      "tags": "",
    "start": 0,
     "genres": "喜剧",
}

headers = {
   "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.101 Safari/537.36"
}

resp = requests.get(url=url, params=params, headers=headers)


print(resp.json())

resp.close()#关注resp
