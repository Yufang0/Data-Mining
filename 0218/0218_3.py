# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 11:30:31 2025

ID:411422259 蕭羽芳
Topic:txt
"""

with open("cardata.txt","r") as f:
    data=[]
    for line in f:
        data.append(line.strip().split(","))
        
import pandas as pd
car = pd.DataFrame(data)
car.columns = ["價格","品質","幾人座","車門","後車箱大小","維修費用","是否推薦"]