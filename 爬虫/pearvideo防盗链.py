#!usr/bin/python3

import requests

#1拿到contId
# 2拿到videoStatus返回的json, -->srcUrl
# 3对src里面的内容进行修改
# 4下载视频


#拉取视频的网址
url = "https://www.pearvideo.com/video_1732235"

contId=url.split("_")[1]

videoStatus=f"https://www.pearvideo.com/videoStatus.jsp?contId={contId}&mrd=0.6305040941739859"

headers={
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.106 Safari/537.36",
    #"Referer": "https://www.pearvideo.com/video_1732235",#防盗链：溯源，本次请求的上一级是谁
    "Referer":url
}


resp=requests.get(videoStatus,headers=headers)

#resp=requests.get(videoStatus)

#print(resp.text)

#print(resp.json())
#print(type(resp.json()))#<class 'dict'>

json_dict =resp.json()
srcUrl=json_dict['videoInfo']['videos']['srcUrl']
systemTime=json_dict['systemTime']

srcUrl=srcUrl.replace(systemTime, f"cont-{contId}")

#下载地址：https://video.pearvideo.com/mp4/adshort/20210617/cont-1732235-15697552_adpkg-ad_hd.mp4
#srcUrl：https://video.pearvideo.com/mp4/adshort/20210617/1623922434371-15697552_adpkg-ad_hd.mp4

#print(srcUrl)



with open("沙漠新土.mp4", mode="wb") as f:
    f.write(requests.get(srcUrl).content)

print("Over!!")
