# pip3 install pandas
import pandas as pd

# 辞書からDataFrameを作成
data = {
    "名前": ["Alice", "Bob", "Charlie", "David"],
    "年齢": [24, 30, 18, 35],
    "点数": [85, 92, 78, 88]
}

df = pd.DataFrame(data)
print(df)

#      名前  年齢  点数
#0   Alice  24  85
#1     Bob  30  92
#2  Charlie  18  78
#3   David  35  88
