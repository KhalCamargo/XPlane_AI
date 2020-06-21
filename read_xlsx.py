import pandas as pd
import numpy as np

xls = pd.read_excel('Dados Tempestade Cirrus.xlsx', sheet_name='Planilha2', usecols = "X,AB")
xls = xls.to_numpy()

erroPitch = xls[:,0]
cmdPitch = xls[:,1]

data = np.column_stack((erroPitch,cmdPitch))

print(data)