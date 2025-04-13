# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 02:40:59 2025

@author: User
"""
import pandas as pd
import xml.etree.ElementTree as ET

tree = ET.parse("read.xml")
root = tree.getroot()

data = []

for d in root:
    sno = d.find("sno").text
    sna = d.find("sna").text
    sarea = d.find("sarea").text
    ar = d.find("ar").text
    tot = d.find("tot").text
    data.append([sno,sna,sarea,ar,tot])
    
data = pd.DataFrame(data)

#是=
data.columns=["sno","sna","sarea","ar","tot"]

#index會忘
data.to_csv("Ubike.csv",index=False)
