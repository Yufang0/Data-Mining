# -*- coding: utf-8 -*-
"""
Spyder Editor
ID:411422259 蕭羽芳
Topic:DataFrame Merge, Concat
"""
import pandas as pd
df1 = pd.DataFrame({
    "A":["A0","A1","A2"],
    "B":["B0","B1","B2"],
    "C":["C0","C1","C2"],},
    index=[0,1,2])

df2 = pd.DataFrame({
    "A":["A3","A4","A5"],
    "B":["B3","B4","B5"],
    "C":["C3","C4","C5"],},
    index=[0,1,2])

result = pd.concat([df1,df2]) #列合併
result2 = pd.concat([df1,df2],axis=1)#欄合併

df3 = pd.DataFrame({
    "Key":["K0","K1","K2"],
    "A":["A0","A1","A2"],
    "B":["B0","B1","B2"],
    "C":["C0","C1","C2"]})

df4 = pd.DataFrame({
    "Key":["K0","K1","K2"],
    "D":["D3","D4","D5"],
    "E":["E3","E4","E5"],
    "F":["F3","F4","F5"]})

merge2 = pd.merge(df3, df4, on="Key")