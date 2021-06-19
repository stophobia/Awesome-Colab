#!usr/bin/python3

import requests

from lxml import etree


xml="""
<book>
    <id>1</id>
    <name>野花遍地香</name>
    <price>1.23</price>
    <nick>白豆腐</nick>
    <author>
        <nick id="10086">周大强</nick>
        <nick id="10010">周芷若</nick>
        <nick class="joy">毛腊肉</nick>
        <nick class="jolin">习包子</nick>
        <div>
            <nick>上海热</nick>
        </div>
        <span>
            <nick>北京瀚</nick>
        </span>
    </author>
</book>
"""

#tree = etree.XML(xml)

#esult = tree.xpath("/book/name")#/表示层级关系，第一个/是根结点
#result = tree.xpath("/book/name/text()")#text()拿文本
#result = tree.xpath("/book/author/nick/text()")#['周大强', '周芷若', '毛腊肉', '习包子']

#result = tree.xpath("/book/author//nick/text()")#['周大强', '周芷若', '毛腊肉', '习包子', '上海热']#//后代
#result = tree.xpath("/book/author/*/nick/text()")#*任意的节点，通配符

#print(result)



text = '''
<li class="li li-first" name="item"><a href="https://ask.hellobi.com/link.html">first item</a></li>
'''
html = etree.HTML(text)
result = html.xpath('//li[contains(@class, "li") and @name="item"]/a/text()')
print(result)







