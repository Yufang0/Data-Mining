# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 10:57:08 2025

ID:411422259 蕭羽芳
Topic:Diabetes
"""

from sklearn.datasets import load_diabetes
suger=load_diabetes()

import pandas as pd
df = pd.DataFrame(data=suger.data,columns=suger.feature_names)
df["target"] = suger.target