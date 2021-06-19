#!usr/bin/python3

import requests
"""
session=requests.session()

data={
    "loginName":"18614075987",
    "password":"q6035945"
}

#1.登录
url = "https://passport.17k.com/ck/user/login"


#resp=session.post(url, data=data)
session.post(url, data=data)
#print(resp.text)
#print(resp.cookies)

#2.拿书架上的cookie，刚才那个session是有cookie的

resp = session.get('https://user.17k.com/ck/author/shelf?page=1&appKey=2406394919')
"""

resp=requests.get('https://user.17k.com/ck/author/shelf?page=1&appKey=2406394919',headers={
    "cookie":"GUID=8d54bb2a-ad9e-4922-b4b4-8e66763bfd4b; c_channel=0; c_csc=web; c_referer_17k=; accessToken=avatarUrl%3Dhttps%253A%252F%252Fcdn.static.17k.com%252Fuser%252Favatar%252F16%252F16%252F64%252F75836416.jpg-88x88%253Fv%253D1610625030000%26id%3D75836416%26nickname%3DIT%25E5%25A4%25A7%25E5%25B8%2588%25E5%2593%25A5%25E6%2598%25AF%25E5%25B8%2585%25E6%25AF%2594%26e%3D1639472429%26s%3Dc755c389cf5013ee"
})
print(resp.text)
#print(resp.json())



