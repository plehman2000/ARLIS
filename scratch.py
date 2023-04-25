import numpy as np

import pandas as pd

df = pd.read_csv("fort.csv")
# print(df)
samples = []
for row in df.iterrows():
    row = row[1]
    print(row)
    sample = {
        "text": row['text_EN'],
        "topics": [row['Topic 1'].split("|")[0], row['Topic 2'].split("|")[0]]
    }
    samples.append(sample)



