#!usr/bin/python3

import requests

proxies = {
    "https":"https://218.60.8.83:3129"
}
url="https://www.baidu.com"
resp=requests.get(url=url, proxies=proxies)
resp.encoding='utf-8'

print(resp.text)
