<<<<<<< HEAD
""" helper code to split the datasets """

import pandas as pd
import numpy as np

np.random.seed(42)

DATA_PATH = "data/HIGGS.csv"

chunksize = 10000

chunk_train = pd.read_csv(DATA_PATH, chunksize=chunksize, skiprows=0, nrows=660000)
chunk_valid = pd.read_csv(DATA_PATH, chunksize=chunksize, skiprows=660000, nrows=220000)
chunk_test = pd.read_csv(DATA_PATH, chunksize=chunksize, skiprows=880000, nrows=220000)

for chunk in chunk_train:
    chunk.to_csv('data/train.csv', sep=',', mode='a')

for chunk in chunk_valid:
    chunk.to_csv('data/valid.csv', sep=',', mode='a')

for chunk in chunk_test:
    chunk.to_csv('data/test.csv', sep=',', mode='a')
=======
""" helper code to split the datasets """

import pandas as pd
import numpy as np

np.random.seed(42)

DATA_PATH = "data/HIGGS.csv"

chunksize = 10000

chunk_train = pd.read_csv(DATA_PATH, chunksize=chunksize, skiprows=0, nrows=660000)
chunk_valid = pd.read_csv(DATA_PATH, chunksize=chunksize, skiprows=660000, nrows=220000)
chunk_test = pd.read_csv(DATA_PATH, chunksize=chunksize, skiprows=880000, nrows=220000)

for chunk in chunk_train:
    chunk.to_csv('data/HIGGS_TRAIN.csv', sep=',', mode='a')

for chunk in chunk_valid:
    chunk.to_csv('data/HIGGS_VALID.csv', sep=',', mode='a')

for chunk in chunk_test:
    chunk.to_csv('data/HIGGS_TEST.csv', sep=',', mode='a')
>>>>>>> 294cd2c1300243e63729d9a751636172d481a951
