# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 09:30:58 2025

Author: 蕭羽芳 統資三乙 411422259
Topic: XML
"""


import xml.etree.ElementTree as ET
import pandas as pd

tree = ET.parse("read.xml")
root = tree.getroot()

data = []
for row in root:
    sno = row.find("sno").text
    sna = row.find("sna").text
    tot = row.find("tot").text
    sarea = row.find("sarea").text
    sareaen = row.find("sareaen").text
    ar = row.find("ar").text
    data.append([sno,sna,tot,sarea,sareaen,ar])
    
df = pd.DataFrame(data)
df.columns=["站編號","站名稱","英文站名","總量數",
            "站區域","站地址"]