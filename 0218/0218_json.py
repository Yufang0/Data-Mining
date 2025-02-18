# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 11:47:45 2025

ID:411422259 蕭羽芳
Topic:JSON
"""

import json
import pandas as pd

with open("pokemon.json","r") as file:
    df = pd.DataFrame(json.load(file))