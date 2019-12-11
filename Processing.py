import numpy as np
import xlrd
import pandas as pd
from sklearn import preprocessing


df = pd.read_excel('wage.xls', header = None)
doc = xlrd.open_workbook('wage.xls').sheet_by_index(0)

attributeNames = doc.row_values(0, 1, 8)
n = len(df.index)
df.reset_index()
df.reindex(index=range(0,n))
df.dropna(inplace=True)
dfMatrix = df.values

y = dfMatrix[1:,0]
yMatrix = np.mat(y)

N = len(y)
M = len(attributeNames)

X = np.mat(np.empty((N,M)))
for i, col_id in enumerate(range(1,M+1)):
    X[:,i] = np.matrix(doc.col_values(col_id, 1, n)).T
    
classX = np.asarray(X)
stdX = preprocessing.scale(classX)

classNames = ['Poor', 'Lower', 'Middle', 'Upper']
classY = np.asarray(np.mat(np.empty((N))).T).squeeze()
for i in range(0,N):
    if y[i] <= np.percentile(y,25):
        classY[i] = 0
    elif y[i] <= np.percentile(y,50):
        classY[i] = 1
    elif y[i] <= np.percentile(y,75):
        classY[i] = 2
    else: 
        classY[i] = 3
        
C = len(classNames)
