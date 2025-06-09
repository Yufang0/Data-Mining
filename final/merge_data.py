# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 16:54:43 2025

@author: merge_data
"""

import pandas as pd

# 設定檔案路徑
file_paths = {
    "v1A": "108年至110年山域意外事故救援案件清冊2024版v1A.csv",
    "v1B": "108年至110年山域意外事故救援案件清冊2024版v1B.csv",
    "111": "111年山域意外事故救援案件清冊(CSV).csv",
    "112": "112年山域意外事故救援案件清冊(CSV).csv",
    "113": "113年山域意外事故救援案件清冊(CSV).csv"
}

# 嘗試不同編碼方式讀檔
def read_file(path):
    for encoding in ["utf-8", "utf-8-sig", "big5", "cp950"]:
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception:
            continue
    raise ValueError(f"無法讀取檔案：{path}")

# 讀取並加入來源欄位
df_v1A = read_file(file_paths["v1A"])
df_v1B = read_file(file_paths["v1B"])
df_111 = read_file(file_paths["111"])
df_112 = read_file(file_paths["112"])
df_113 = read_file(file_paths["113"])

# 合併所有資料
df_all = pd.concat([df_v1A, df_v1B, df_111, df_112, df_113], ignore_index=True)

# 去除完全重複的列
df_cleaned = df_all.drop_duplicates()

# 去除第一筆資料（如果那是欄位名稱的一部分）
df_cleaned = df_cleaned.iloc[1:].reset_index(drop=True)

# 統一欄位格式：轉為字串後 strip 空白與換行
df_cleaned["year"] = df_cleaned["year"].astype(str).str.strip()
df_cleaned["District fire protection unit"] = df_cleaned["District fire protection unit"].astype(str).str.strip()
df_cleaned["Volume-Total"] = df_cleaned["Volume-Total"].astype(str).str.strip()
df_cleaned["Number of police dispatches"] = df_cleaned["Number of police dispatches"].astype(str).str.strip()
df_cleaned["National Park Service missions"] = df_cleaned["National Park Service missions"].astype(str).str.strip()
df_cleaned["Number of dispatches from the Forest Service"] = df_cleaned["Number of dispatches from the Forest Service"].astype(str).str.strip()
df_cleaned["Person"] = df_cleaned["Person"].astype(str).str.strip()
df_cleaned["Ministry of Defense Seagull helicopter sorties"] = df_cleaned["Ministry of Defense Seagull helicopter sorties"].astype(str).str.strip()
df_cleaned["Quantity"] = df_cleaned["Quantity"].astype(str).str.strip()
df_cleaned["Search and rescue dog dispatches"] = df_cleaned["Search and rescue dog dispatches"].astype(str).str.strip()
df_cleaned["Unmanned aerial photography sorties dispatched"] = df_cleaned["Unmanned aerial photography sorties dispatched"].astype(str).str.strip()
df_cleaned["death"] = df_cleaned["death"].astype(str).str.strip()
df_cleaned["rescued"] = df_cleaned["rescued"].astype(str).str.strip()
df_cleaned["health status"] = df_cleaned["health status"].astype(str).str.strip()
df_cleaned["missing"] = df_cleaned["missing"].astype(str).str.strip()
# 去除前後空白、換行符號
df_cleaned.columns = df_cleaned.columns.str.strip().str.replace("\n", "").str.replace("\r", "")


import pandas as pd

# 複製原始資料（假設 df_cleaned 是合併後的完整資料）
df_time = df_cleaned.copy()

# 指定欄位名稱（依實際欄位名稱調整）
col_report = "Reporting date"
col_rescue = "Search and rescue time"
col_close = "Case closing time"


# 轉換為 datetime 格式
df_time["報案時間"] = pd.to_datetime(df_time[col_report], errors="coerce")
df_time["搜救時間"] = pd.to_datetime(df_time[col_rescue], errors="coerce")
#df_time["結案時間"] = pd.to_datetime(df_time[col_close], errors="coerce")

# 計算等待與救援時長（以小時為單位）
df_time["救援時長(分鐘)"] = (df_time["搜救時間"] - df_time["報案時間"]).dt.total_seconds() / 60
#df_time["實際救援時長(分鐘)"] = (df_time["結案時間"] - df_time["搜救時間"]).dt.total_seconds() / 60

# 如果你要丟回 df_cleaned：
df_cleaned["RescueTime_Min"] = df_time["救援時長(分鐘)"]
#df_cleaned["RescueTime_Min"] = df_time["實際救援時長(分鐘)"]
df_cleaned.info()
# 統計每個欄位的值分布（前10個最常見的值）
for col in df_cleaned.columns:
    print(f"\n=== 欄位：{col} ===")
    print(df_cleaned[col].value_counts(dropna=False).head(10))

# 另存成 CSV 檔案：
#df_cleaned.to_csv("108-113山域意外事故救援案件.csv", index=False)

#Search and rescue time Nan:112
#Case closing time NaN:1266
#Mountain area management agency (山域管理(制)機關)  NaN:1312 刪除
#Entrance to the mountain NaN:63 刪除
#destinationg NaN:97 刪除
#Out of the mountain pass NaN:76 刪除
#X_3826、Y_3826 刪除
#District fire protection unit 0:145
#Volume-Total 0:2274
#Number of Police Dispatches 0:2233
#National Park Service missions 0:2817 刪除 >2781
#Number of dispatches from the Forest Service 0:2782 刪除 >2781
#Person 0:2957 刪除 >2781
#Ministry of Defense Seagull helicopter sorties 0:3445 >2781
#Quantity 0:2993 刪除 >2781
#Search and rescue dog dispatches 0:3442 刪除 >2781
#Unmanned aerial photography sorties dispatched 0:3419 刪除 >2781 (8成)
# 刪除指定欄位
cols_to_drop = [
    "Reporting date", "Search and rescue time", "Case closing time",
    "Mountain area management agency", "Entrance to the mountain",
    "destination", "Out of the mountain pass",
    "X_3826", "Y_3826",
    "National Park Service missions",
    "Number of dispatches from the Forest Service",
    "Person", "Ministry of Defense Seagull helicopter sorties", "Quantity",
    "Search and rescue dog dispatches",
    "Unmanned aerial photography sorties dispatched"
]

df_cleaned = df_cleaned.drop(columns=cols_to_drop, errors="ignore")

# 指定要統計的欄位
cols = ["death", "rescued", "health status", "missing"]

# 直接使用原始欄位，不轉型，保留 NaN
combination_counts = df_cleaned[cols].groupby(cols, dropna=False).size().reset_index(name="count")

# 排序
combination_counts = combination_counts.sort_values(by="count", ascending=False)

# 顯示結果
#print(combination_counts)


# 複製資料避免影響原始 DataFrame
df_labeled = df_cleaned.copy()

# 將條件欄位轉為數值（若還未轉型）
for col in ["death", "rescued", "health status", "missing"]:
    df_labeled[col] = pd.to_numeric(df_labeled[col], errors="coerce")

# 建立分類欄位 outcome_label
def classify_outcome(row):
    if (
        ((row["rescued"] > 0) or (row["health status"] > 0)) and
        (row["death"] == 0) and
        (row["missing"] == 0)
    ):
        return "rescued"
    elif (
        (row["rescued"] == 0) and
        (row["health status"] == 0) and
        (row["death"] > 0) and
        (row["missing"] == 0)
    ):
        return "death"
    elif (
        (row["rescued"] == 0) and
        (row["health status"] == 0) and
        (row["death"] == 0) and
        (row["missing"] > 0)
    ):
        return "missing"
    else:
        return "Other"

# 套用分類
df_labeled["outcome_label"] = df_labeled.apply(classify_outcome, axis=1)

# 檢查分類結果
#print(df_labeled["outcome_label"].value_counts())

# 複製資料
df_final = df_labeled.copy()

# 建立新欄位 'rescued'，根據 outcome_label 對應轉換為 Yes/No/Other
df_final["rescued"] = df_final["outcome_label"].replace({
    "rescued": "Yes",
    "death": "No",
    "missing": "No"
}).fillna("Other")  # 其他情況歸為 Other

# 刪除原始四個欄位與 outcome_label
df_final = df_final.drop(columns=["death", "health status", "missing", "outcome_label"])
# 檢查分類結果
#print(df_final["rescued"].value_counts())
'''
# 過濾 outcome_label 為 "其他" 的資料
df_other = df_labeled[df_labeled["outcome_label"] == "其他"]

# 指定要分析的四個欄位
cols = ["death", "rescued", "health status", "missing"]

# 統計每種組合出現的次數（包含 NaN）
other_combinations = df_other.groupby(cols, dropna=False).size().reset_index(name="count")

# 按照 count 遞減排序
other_combinations = other_combinations.sort_values(by="count", ascending=False)

# 顯示結果
print(other_combinations)
'''
#Cases older than 48 hours (yes/no)   nan刪除 192
#Mountain domain name nan刪除 2
#Main cause of occurrence nan刪除 40
#Whether it violates relevant mountain climbing laws 補值 255
#death nan刪除 33
#rescued nan刪除 33
#health status nan刪除 33
#missing nan刪除 33
#Yes/no Send to hospital nan刪除 6
#RescueTime_Min 取決 nan:1014

# 指定需要刪除 NaN 的欄位
cols_to_dropna = [
    "Cases older than 48 hours (yes/no)",
    "Mountain domain name",
    "Main cause of occurrence",
    "Yes/no Send to hospital"
]

# 刪除上述欄位中含有 NaN 的列
df_final = df_final.dropna(subset=cols_to_dropna)
# 保留 rescued 欄位不是 "Other" 的資料
df_final = df_final[df_final["rescued"] != "Other"]
df_final.info()


    

# 每年每欄位的缺失值數量
missing_by_year = df_final.groupby("year").apply(lambda g: g.isna().sum())

# 顯示結果
print(missing_by_year)