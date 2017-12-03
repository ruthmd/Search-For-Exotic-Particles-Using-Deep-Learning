""" helper code to split the datasets """

import pandas as pd
import numpy as np

np.random.seed(42)

DATA_PATH = "data/HIGGS.csv"

chunksize = 10 ** 6
dt = dict()
for i in range(0,29):
    dt[i] = np.float64

chunk_train = pd.read_csv(DATA_PATH, chunksize=chunksize, skiprows=0, nrows=660000, header=None, dtype=dt) # 660000
chunk_valid = pd.read_csv(DATA_PATH, chunksize=chunksize, skiprows=660000, nrows=220000, header=None, dtype=dt)
chunk_test = pd.read_csv(DATA_PATH, chunksize=chunksize, skiprows=880000, nrows=220000, header=None, dtype=dt)

for chunk in chunk_train:
    chunk.to_csv('data/train.csv', sep=',', mode='a', index=False)

for chunk in chunk_valid:
    chunk.to_csv('data/valid.csv', sep=',', mode='a', index=False)

for chunk in chunk_test:
    chunk.to_csv('data/test.csv', sep=',', mode='a', index=False)
