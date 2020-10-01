# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 12:26:16 2020

@author: Pei-yuChen
"""

import pandas as pd

reader = pd.read_csv("reader.csv", index_col=0)
emobank = pd.read_csv("emobank.csv", index_col=0)

common = sorted(list(set(reader.index).intersection(set(emobank.index))))


reader_x = (reader.loc[common, "V"]).values
text_x = (emobank.loc[common, "text"]).values
split_x = (emobank.loc[common, "split"]).values

 
d = {'text': text_x,'split': split_x, 'V': reader_x}
reader = pd.DataFrame(d)

reader.to_csv("reader.csv", index = False)


