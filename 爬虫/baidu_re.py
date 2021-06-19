#!/usr/bin/python3

import requests  as re

url = 'http://guoxue.httpcn.com/book/mengzi/'

resp = re.request('GET', url=url)
text = resp.content

with open('mybaidu.html', 'wb') as f:
    f.write(text)

print("Over!")



