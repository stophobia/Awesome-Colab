#!usr/bin/python3

import requests
from lxml import etree

params={
    "type":"new",
    "kw":"saas",
}
headers = {
   "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.101 Safari/537.36"
}

url="https://beijing.zbj.com/search/f/?type=new&kw=saas"

#url="https://www.zbj.com/search/f/"

#response = requests.get(url=url,params=params)
response = requests.get(url=url, headers=headers)
response.encoding="utf-8"
print(response.url)

#解析
html = etree.HTML(response.text)
#result = etree.tostring(html)
#print(result.decode('utf-8'))


#拿到每一个服务商的div
divs = html.xpath("/html/body/div[6]/div/div/div[2]/div[5]/div/div")
#divs = html.xpath("/html/body/div[6]/div/div/div[2]/div[5]/div[1]/div")
#//*[@id="utopia_widget_70"]/a[1]/div[2]/div[1]/span[1]
#divs = html.xpath("/html/body/div[6]/div/div/div[2]/div[5]/div[1]/div[1]")
#print(divs)
for div in divs:
    price=div.xpath("./div/div/a[1]/div[2]/div[1]/span[1]/text()")[0].strip("¥")
    title="saas".join(div.xpath("./div/div/a[1]/div[2]/div[2]/p/text()"))
    com_name=div.xpath("./div/div/a[2]/div[1]/p/text()")[-1]
    com_address=div.xpath("./div/div/a[2]/div[1]/div/span/text()")[0]
    print(title)
    print(price)
    print(com_name)
    print(com_address)

#/html/body/div[6]/div/div/div[2]/div[5]/div/div[1]/div/div/a[2]/div[1]/div/span
#/html/body/div[6]/div/div/div[2]/div[5]/div/div[1]/div/div/a[2]/div[1]/div/span
#/html/body/div[6]/div/div/div[2]/div[5]/div/div[1]/div/div/a[2]/div[1]/div/span
#/html/body/div[6]/div/div/div[2]/div[5]/div[1]/div[1]/div/div/a[2]/div[1]/p/text()