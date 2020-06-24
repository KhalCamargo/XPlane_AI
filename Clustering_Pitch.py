import numpy as np
import skfuzzy as fuzz
from matplotlib import pyplot as plt
import pandas as pd
import skfuzzy.control as ctrl

# Gathering all data
data = pd.read_excel('C:\\Workspace\\XPlane\\Dados USAR SO PRA PITCH 3.xlsx')
data = data.drop(data.index[0:159]) #Removing n-first rows

data2 = pd.read_excel('C:\\Workspace\\XPlane\\Dados USAR SO PRA PITCH 2.xlsx')
data2 = data.drop(data.index[0:284]) #Removing n-first rows

data3 = pd.read_excel('C:\\Workspace\\XPlane\\Dados USAR SO PRA PITCH.xlsx')
data3 = data.drop(data.index[0:1003]) #Removing n-first rows

# Appending data
data = data.append(data2)
data = data.append(data3)

data = data.loc[(abs(data['   pitch,__deg '])) <= 15] #Selecting only pitch within range
data = data.loc[:,'ErroPitch':]


Erro_Cmd_Pitch = data.loc[:,['ErroPitch','CmdPitch']]
Erro_Cmd_Pitch = Erro_Cmd_Pitch.to_numpy()

dErro_Cmd_Pitch = data.loc[:,['dErroP','CmdPitch']]
dErro_Cmd_Pitch = dErro_Cmd_Pitch.to_numpy()


xlsErro = Erro_Cmd_Pitch
xlsdErro = dErro_Cmd_Pitch

erro = xlsErro[:,0]
cmd = xlsErro[:,1]

derro = xlsdErro[:,0]
cmddErro = xlsdErro[:,1]


data = np.column_stack((erro,cmd))
data2 = np.column_stack((derro,cmddErro))
n_clusters = 7

cntrError, uError, u0Error, dError, jmError, pError, fpcError = fuzz.cluster.cmeans(data.T, c=n_clusters, m=2, error=0.005, maxiter=1000, init=None)
cntrdError, udError, u0dError, ddError, jmdError, pdError, fpcdError = fuzz.cluster.cmeans(data2.T, c=n_clusters, m=2, error=0.005, maxiter=1000, init=None)
'''
inputs:
n = numero de clusters
m = valor  maior que 1
error = criterio de parada
metric = euclidiana
outputs:
cntr= centro de cada cluster
u = matriz de saida com graus dew pertinencia
'''
cluster_membership = np.argmax(udError, axis=0)

fig3, ax3 = plt.subplots()
ax3.set_title('Trained Model')
for j in range(n_clusters):
    ax3.plot(data2[cluster_membership == j, 0],
             data2[cluster_membership == j, 1], 'o',
             label='series ' + str(j))

for pt in cntrdError:
    ax3.plot(pt[0], pt[1], 'rs')

ax3.legend()
plt.show()

""" Controlador:
NL-negative large
NM-negative medium
NS-negative small
AZ-approximately zero
PS-positive small
PM-positive medium
PL-positive large
"""

# Create the three fuzzy variables - two inputs, one output
ErrorPitch = ctrl.Antecedent(np.arange(-16, 16, 0.001), 'Error - Pitch')
dErrorPitch = ctrl.Antecedent(np.arange(-5, 5, 0.001), 'Derivative Error - Pitch')
ElevStick = ctrl.Consequent(np.arange(-0.3, 0.3, 0.01), 'Elevation Stick')

# Auto-membership function population is possible with .automf(3, 5, or 7)

ElevStick.automf(3)

cntrauxError = []
for i in range (0,7):
    cntrauxError.append(cntrError[i,0])
cntrauxError.sort()

# Custom membership functions can be built interactively with a familiar,
ErrorPitch['NL'] = fuzz.trapmf(ErrorPitch.universe, [-90, -89, cntrauxError[0], cntrauxError[1]])
ErrorPitch['NM'] = fuzz.trimf(ErrorPitch.universe, [cntrauxError[0], cntrauxError[1], cntrauxError[2]])
ErrorPitch['NS'] = fuzz.trimf(ErrorPitch.universe, [cntrauxError[1], cntrauxError[2], cntrauxError[3]])
ErrorPitch['AZ'] = fuzz.trimf(ErrorPitch.universe, [cntrauxError[2], cntrauxError[3], cntrauxError[4]])
ErrorPitch['PS'] = fuzz.trimf(ErrorPitch.universe, [cntrauxError[3], cntrauxError[4], cntrauxError[5]])
ErrorPitch['PM'] = fuzz.trimf(ErrorPitch.universe, [cntrauxError[4], cntrauxError[5], cntrauxError[6]])
ErrorPitch['PL'] = fuzz.trapmf(ErrorPitch.universe, [cntrauxError[5], cntrauxError[6], 89,90])

cntrauxdError = []
for i in range (0,7):
    cntrauxdError.append(cntrdError[i,0])
cntrauxdError.sort()

dErrorPitch['NL'] = fuzz.trapmf(dErrorPitch.universe, [-90, -89, cntrauxdError[0], cntrauxdError[1]])
dErrorPitch['NM'] = fuzz.trimf(dErrorPitch.universe, [cntrauxdError[0], cntrauxdError[1], cntrauxdError[2]])
dErrorPitch['NS'] = fuzz.trimf(dErrorPitch.universe, [cntrauxdError[1], cntrauxdError[2], cntrauxdError[3]])
dErrorPitch['AZ'] = fuzz.trimf(dErrorPitch.universe, [cntrauxdError[2], cntrauxdError[3], cntrauxdError[4]])
dErrorPitch['PS'] = fuzz.trimf(dErrorPitch.universe, [cntrauxdError[3], cntrauxdError[4], cntrauxdError[5]])
dErrorPitch['PM'] = fuzz.trimf(dErrorPitch.universe, [cntrauxdError[4], cntrauxdError[5], cntrauxdError[6]])
dErrorPitch['PL'] = fuzz.trapmf(dErrorPitch.universe, [cntrauxdError[5], cntrauxdError[6], 89,90])


ErrorPitch.view()
dErrorPitch.view()
ElevStick.view()
plt.show()
