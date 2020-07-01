import numpy as np
import skfuzzy as fuzz
from matplotlib import pyplot as plt
import pandas as pd
import skfuzzy.control as ctrl

def getData():
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

    ##### New Code
    datas = data.loc[:,['ErroPitch','dErroP','CmdPitch']]

    #datas.loc[:,'ErroPitch':'dErroP'] = (datas.loc[:,'ErroPitch':'dErroP'] - datas.loc[:,'ErroPitch':'dErroP'].min())/(datas.loc[:,'ErroPitch':'dErroP'].max()-datas.loc[:,'ErroPitch':'dErroP'].min())
    return datas

def normalizeDatas(datas):
    datas = (datas - datas.min())/(datas.max()-datas.min())
    return datas
def normalizeNewData(data,datas):    
    toClassDF = {'ErroPitch':[data[0][0]],
                 'dErroP':[data[0][1]]}
    toClass = pd.DataFrame(toClassDF,columns=['ErroPitch','dErroP'])
    toClass = (toClass -  datas.min())/(datas.max()-datas.min())
    if toClass > 1:
        toClass = 1
    if toClass < 0:
        toClass = 0
    return toClass.to_numpy()
def generateFuzzySys(datas):
    datasnp = datas.to_numpy()
    n_clusters = 7
    cntrError, uError, u0Error, dError, jmError, pError, fpcError = fuzz.cluster.cmeans(datasnp.T, c=n_clusters, m=2, error=0.005, maxiter=1000, init=None)

    cntrStick = []
    for i in range (0,n_clusters):
        cntrStick.append(cntrError[i,2])
    cntrStick.sort()

    ElevStick = ctrl.Consequent(np.arange(-1, 1.001, 0.001), 'Elevation Stick')
    """ Saída Stick
    SL - Subir Large
    SM - Subir Medium
    SS - Subir Small
    M - Manter
    DS - Descer Small
    DM - Descer Medium
    DL - Descer Large
    """
    ElevStick['DL'] = fuzz.trapmf(ElevStick.universe,[-2,-1,cntrStick[0],cntrStick[1]])
    ElevStick['DM'] = fuzz.trimf(ElevStick.universe,[cntrStick[0],cntrStick[1],cntrStick[2]])
    ElevStick['DS'] = fuzz.trimf(ElevStick.universe,[cntrStick[1],cntrStick[2],cntrStick[3]])
    ElevStick['M'] = fuzz.trimf(ElevStick.universe,[cntrStick[2],cntrStick[3],cntrStick[4]])
    ElevStick['SS'] = fuzz.trimf(ElevStick.universe,[cntrStick[3],cntrStick[4],cntrStick[5]])
    ElevStick['SM'] = fuzz.trimf(ElevStick.universe,[cntrStick[4],cntrStick[5],cntrStick[6]])
    ElevStick['SL'] = fuzz.trapmf(ElevStick.universe,[cntrStick[5],cntrStick[6],1,2])

 

    ## New Rules
    cntrErrorP = []
    for i in range (0,n_clusters):
        cntrErrorP.append(cntrError[i,0])
    cntrErrorP.sort()

    Error = ctrl.Antecedent(np.arange(0,1.001,0.001),'Error')
    """ Entrada Cluster
    NL-negative large
    NM-negative medium
    NS-negative small
    AZ-approximately zero
    PS-positive small
    PM-positive medium
    PL-positive large
    """
    Error['NL'] = fuzz.trapmf(Error.universe,[-2,-1,cntrErrorP[0],cntrErrorP[1]])
    Error['NM'] = fuzz.trimf(Error.universe,[cntrErrorP[0],cntrErrorP[1],cntrErrorP[2]])
    Error['NS'] = fuzz.trimf(Error.universe,[cntrErrorP[1],cntrErrorP[2],cntrErrorP[3]])
    Error['Z'] = fuzz.trimf(Error.universe,[cntrErrorP[2],cntrErrorP[3],cntrErrorP[4]])
    Error['PS'] = fuzz.trimf(Error.universe,[cntrErrorP[3],cntrErrorP[4],cntrErrorP[5]])
    Error['PM'] = fuzz.trimf(Error.universe,[cntrErrorP[4],cntrErrorP[5],cntrErrorP[6]])
    Error['PL'] = fuzz.trapmf(Error.universe,[cntrErrorP[5],cntrErrorP[6],1,2])

    cntrDErrorP = []
    for i in range (0,n_clusters):
        cntrDErrorP.append(cntrError[i,1])
    cntrDErrorP.sort()

    dError = ctrl.Antecedent(np.arange(0,1.001,0.001),'Error Derivative')

    dError['NL'] = fuzz.trapmf(dError.universe,[-2,-1,cntrDErrorP[0],cntrDErrorP[1]])
    dError['NM'] = fuzz.trimf(dError.universe,[cntrDErrorP[0],cntrDErrorP[1],cntrDErrorP[2]])
    dError['NS'] = fuzz.trimf(dError.universe,[cntrDErrorP[1],cntrDErrorP[2],cntrDErrorP[3]])
    dError['Z'] = fuzz.trimf(dError.universe,[cntrDErrorP[2],cntrDErrorP[3],cntrDErrorP[4]])
    dError['PS'] = fuzz.trimf(dError.universe,[cntrDErrorP[3],cntrDErrorP[4],cntrDErrorP[5]])
    dError['PM'] = fuzz.trimf(dError.universe,[cntrDErrorP[4],cntrDErrorP[5],cntrDErrorP[6]])
    dError['PL'] = fuzz.trapmf(dError.universe,[cntrDErrorP[5],cntrDErrorP[6],1,2])

    allRules = list()

    # Regras dos extremos
    #R1 = ctrl.Rule(Error['NL'],ElevStick['SL'])
    #allRules.append(R1)
    #R2 = ctrl.Rule(Error['PL'],ElevStick['DL'])
    #allRules.append(R2)
    R3 = ctrl.Rule(((Error['NM'] | Error['NS']) & (dError['NL'] | dError['NM'])) | (Error['Z'] & dError['NL']) | (Error['NM'] & dError['NS'] | Error['NL']),ElevStick['SL'])
    allRules.append(R3)
    R4 = ctrl.Rule(((Error['PM'] | Error['PS']) & (dError['PL'] | dError['PM'])) | (Error['Z'] & dError['PL']) | (Error['PM'] & dError['PS'] | Error['PL']),ElevStick['DL'])
    allRules.append(R4)
    R5 = ctrl.Rule((Error['PS'] & dError['NL']) | (Error['Z'] & dError['NM']) | (Error['NS'] & dError['NS']) | (Error['NM'] & dError['Z']),ElevStick['SM'])
    allRules.append(R5)
    R6 = ctrl.Rule((Error['PM'] & dError['NL']) | (Error['PS'] & dError['NM']) | (Error['Z'] & dError['NS']) | (Error['NS'] & dError['Z']) | (Error['NM'] & dError['PS']),ElevStick['SS'])
    allRules.append(R6)
    R7 = ctrl.Rule((Error['PM'] & dError['NM']) | (Error['PS'] & dError['NS']) | (Error['Z'] & dError['Z']) | (Error['NS'] & dError['PS']) | (Error['NM'] & dError['PM']),ElevStick['M'])
    allRules.append(R7)
    R8 = ctrl.Rule((Error['PM'] & dError['NS']) | (Error['PS'] & dError['Z']) | (Error['Z'] & dError['PS']) | (Error['NS'] & dError['PM']) | (Error['NM'] & dError['PL']),ElevStick['DS'])
    allRules.append(R8)
    R9 = ctrl.Rule((Error['PM'] & dError['Z']) | (Error['PS'] & dError['PS']) | (Error['Z'] & dError['PM']) | (Error['NS'] & dError['PL']),ElevStick['DM'])
    allRules.append(R9)

    sys = ctrl.ControlSystem(allRules)
    sim = ctrl.ControlSystemSimulation(sys)
    return sim


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

##### New Code
datas = data.loc[:,['ErroPitch','dErroP','CmdPitch']]

datas.loc[:,'ErroPitch':'dErroP'] = (datas.loc[:,'ErroPitch':'dErroP'] - datas.loc[:,'ErroPitch':'dErroP'].min())/(datas.loc[:,'ErroPitch':'dErroP'].max()-datas.loc[:,'ErroPitch':'dErroP'].min())

datasnp = datas.to_numpy()
n_clusters = 7
cntrError, uError, u0Error, dError, jmError, pError, fpcError = fuzz.cluster.cmeans(datasnp.T, c=n_clusters, m=2, error=0.005, maxiter=1000, init=None)
##### End New Code

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
cluster_membership = np.argmax(uError, axis=0)


fig3 = plt.figure()
ax3 = fig3.add_subplot(111, projection='3d')

ax3.set_title('Trained Model')


for j in range(n_clusters):
    ax3.scatter(datasnp[cluster_membership == j, 0],
             datasnp[cluster_membership == j, 1],datasnp[cluster_membership == j, 2], 'o',
             label='series ' + str(j),alpha=0.1)


for pt in cntrError:
    ax3.scatter(pt[0], pt[1],pt[2], marker='s',c='r',s=15**2)

## Create 2D cluster
fig4,axs = plt.subplots(3,1)

for pt in cntrError:
    for j in range(0,3):
        axs[j].plot(pt[j],1,'ro')

for j in range(0,3):
    axs[j].set_ylim([0,1.02])

axs[0].set_title("Pertinencia Unitária para Erro")
axs[1].set_title("Pertinencia Unitária para Derivada do Erro")
axs[2].set_title("Pertinencia Unitária para Saída")
## END Create 2D Cluster


## Predict for new data

#toClass = [[0,0]] #sample data to be analyzed (classified)
#toClassDF = {'ErroPitch':[toClass[0][0]],
#             'dErroP':[toClass[0][1]]}
#toClass = pd.DataFrame(toClassDF,columns=['ErroPitch','dErroP'])
#datas = data.loc[:,['ErroPitch','dErroP','CmdPitch']]
#datas = datas.loc[:,['ErroPitch','dErroP']]
#toClass = (toClass -  datas.min())/(datas.max()-datas.min()) #Normalizing
## if data bigger than max -> saturate
#toClass = toClass.to_numpy()

#ax4.plot(toClass[0][0],toClass[0][1],'bx')

#u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
#    toClass.T,cntrError[:,0:2],m=2, error=0.005, maxiter=1000)

## END Predict for new data
ax3.legend()
plt.show()

cntrStick = []
for i in range (0,n_clusters):
    cntrStick.append(cntrError[i,2])
cntrStick.sort()

ElevStick = ctrl.Consequent(np.arange(-1, 1.001, 0.001), 'Elevation Stick')
""" Saída Stick
SL - Subir Large
SM - Subir Medium
SS - Subir Small
M - Manter
DS - Descer Small
DM - Descer Medium
DL - Descer Large
"""
ElevStick['DL'] = fuzz.trapmf(ElevStick.universe,[-2,-1,cntrStick[0],cntrStick[1]])
ElevStick['DM'] = fuzz.trimf(ElevStick.universe,[cntrStick[0],cntrStick[1],cntrStick[2]])
ElevStick['DS'] = fuzz.trimf(ElevStick.universe,[cntrStick[1],cntrStick[2],cntrStick[3]])
ElevStick['M'] = fuzz.trimf(ElevStick.universe,[cntrStick[2],cntrStick[3],cntrStick[4]])
ElevStick['SS'] = fuzz.trimf(ElevStick.universe,[cntrStick[3],cntrStick[4],cntrStick[5]])
ElevStick['SM'] = fuzz.trimf(ElevStick.universe,[cntrStick[4],cntrStick[5],cntrStick[6]])
ElevStick['SL'] = fuzz.trapmf(ElevStick.universe,[cntrStick[5],cntrStick[6],1,2])

#Play
#Cluster1 = ctrl.Antecedent(np.mgrid[-1:1.001:0.001,-1:1.001:0.001],'Cluster1')

#Rule1 = ctrl.Rule(Cluster1,ElevStick['M'] )
#sys = ctrl.ControlSystem([Rule1])
#sim = ctrl.ControlSystemSimulation(sys)
#

## New Rules
cntrErrorP = []
for i in range (0,n_clusters):
    cntrErrorP.append(cntrError[i,0])
cntrErrorP.sort()

Error = ctrl.Antecedent(np.arange(0,1.001,0.001),'Error')
""" Entrada Cluster
NL-negative large
NM-negative medium
NS-negative small
AZ-approximately zero
PS-positive small
PM-positive medium
PL-positive large
"""
Error['NL'] = fuzz.trapmf(Error.universe,[-2,-1,cntrErrorP[0],cntrErrorP[1]])
Error['NM'] = fuzz.trimf(Error.universe,[cntrErrorP[0],cntrErrorP[1],cntrErrorP[2]])
Error['NS'] = fuzz.trimf(Error.universe,[cntrErrorP[1],cntrErrorP[2],cntrErrorP[3]])
Error['Z'] = fuzz.trimf(Error.universe,[cntrErrorP[2],cntrErrorP[3],cntrErrorP[4]])
Error['PS'] = fuzz.trimf(Error.universe,[cntrErrorP[3],cntrErrorP[4],cntrErrorP[5]])
Error['PM'] = fuzz.trimf(Error.universe,[cntrErrorP[4],cntrErrorP[5],cntrErrorP[6]])
Error['PL'] = fuzz.trapmf(Error.universe,[cntrErrorP[5],cntrErrorP[6],1,2])

cntrDErrorP = []
for i in range (0,n_clusters):
    cntrDErrorP.append(cntrError[i,1])
cntrDErrorP.sort()

dError = ctrl.Antecedent(np.arange(0,1.001,0.001),'Error Derivative')

dError['NL'] = fuzz.trapmf(dError.universe,[-2,-1,cntrDErrorP[0],cntrDErrorP[1]])
dError['NM'] = fuzz.trimf(dError.universe,[cntrDErrorP[0],cntrDErrorP[1],cntrDErrorP[2]])
dError['NS'] = fuzz.trimf(dError.universe,[cntrDErrorP[1],cntrDErrorP[2],cntrDErrorP[3]])
dError['Z'] = fuzz.trimf(dError.universe,[cntrDErrorP[2],cntrDErrorP[3],cntrDErrorP[4]])
dError['PS'] = fuzz.trimf(dError.universe,[cntrDErrorP[3],cntrDErrorP[4],cntrDErrorP[5]])
dError['PM'] = fuzz.trimf(dError.universe,[cntrDErrorP[4],cntrDErrorP[5],cntrDErrorP[6]])
dError['PL'] = fuzz.trapmf(dError.universe,[cntrDErrorP[5],cntrDErrorP[6],1,2])

allRules = list()

# Regras dos extremos
#R1 = ctrl.Rule(Error['NL'],ElevStick['SL'])
#allRules.append(R1)
#R2 = ctrl.Rule(Error['PL'],ElevStick['DL'])
#allRules.append(R2)
R3 = ctrl.Rule(((Error['NM'] | Error['NS']) & (dError['NL'] | dError['NM'])) | (Error['Z'] & dError['NL']) | (Error['NM'] & dError['NS'] | Error['NL']),ElevStick['SL'])
allRules.append(R3)
R4 = ctrl.Rule(((Error['PM'] | Error['PS']) & (dError['PL'] | dError['PM'])) | (Error['Z'] & dError['PL']) | (Error['PM'] & dError['PS'] | Error['PL']),ElevStick['DL'])
allRules.append(R4)
R5 = ctrl.Rule((Error['PS'] & dError['NL']) | (Error['Z'] & dError['NM']) | (Error['NS'] & dError['NS']) | (Error['NM'] & dError['Z']),ElevStick['SM'])
allRules.append(R5)
R6 = ctrl.Rule((Error['PM'] & dError['NL']) | (Error['PS'] & dError['NM']) | (Error['Z'] & dError['NS']) | (Error['NS'] & dError['Z']) | (Error['NM'] & dError['PS']),ElevStick['SS'])
allRules.append(R6)
R7 = ctrl.Rule((Error['PM'] & dError['NM']) | (Error['PS'] & dError['NS']) | (Error['Z'] & dError['Z']) | (Error['NS'] & dError['PS']) | (Error['NM'] & dError['PM']),ElevStick['M'])
allRules.append(R7)
R8 = ctrl.Rule((Error['PM'] & dError['NS']) | (Error['PS'] & dError['Z']) | (Error['Z'] & dError['PS']) | (Error['NS'] & dError['PM']) | (Error['NM'] & dError['PL']),ElevStick['DS'])
allRules.append(R8)
R9 = ctrl.Rule((Error['PM'] & dError['Z']) | (Error['PS'] & dError['PS']) | (Error['Z'] & dError['PM']) | (Error['NS'] & dError['PL']),ElevStick['DM'])
allRules.append(R9)

sys = ctrl.ControlSystem(allRules)
sim = ctrl.ControlSystemSimulation(sys)
## END New Rules

#""" Controlador:
#NL-negative large
#NM-negative medium
#NS-negative small
#AZ-approximately zero
#PS-positive small
#PM-positive medium
#PL-positive large
#"""

## Create the three fuzzy variables - two inputs, one output
#ErrorPitch = ctrl.Antecedent(np.arange(-16, 16, 0.001), 'Error - Pitch')
#dErrorPitch = ctrl.Antecedent(np.arange(-5, 5, 0.001), 'Derivative Error - Pitch')
#ElevStick = ctrl.Consequent(np.arange(-0.3, 0.3, 0.01), 'Elevation Stick')

## Auto-membership function population is possible with .automf(3, 5, or 7)

#ElevStick.automf(3)

#cntrauxError = []
#for i in range (0,7):
#    cntrauxError.append(cntrError[i,0])
#cntrauxError.sort()

## Custom membership functions can be built interactively with a familiar,
#ErrorPitch['NL'] = fuzz.trapmf(ErrorPitch.universe, [-90, -89, cntrauxError[0], cntrauxError[1]])
#ErrorPitch['NM'] = fuzz.trimf(ErrorPitch.universe, [cntrauxError[0], cntrauxError[1], cntrauxError[2]])
#ErrorPitch['NS'] = fuzz.trimf(ErrorPitch.universe, [cntrauxError[1], cntrauxError[2], cntrauxError[3]])
#ErrorPitch['AZ'] = fuzz.trimf(ErrorPitch.universe, [cntrauxError[2], cntrauxError[3], cntrauxError[4]])
#ErrorPitch['PS'] = fuzz.trimf(ErrorPitch.universe, [cntrauxError[3], cntrauxError[4], cntrauxError[5]])
#ErrorPitch['PM'] = fuzz.trimf(ErrorPitch.universe, [cntrauxError[4], cntrauxError[5], cntrauxError[6]])
#ErrorPitch['PL'] = fuzz.trapmf(ErrorPitch.universe, [cntrauxError[5], cntrauxError[6], 89,90])

#cntrauxdError = []
#for i in range (0,7):
#    cntrauxdError.append(cntrdError[i,0])
#cntrauxdError.sort()

#dErrorPitch['NL'] = fuzz.trapmf(dErrorPitch.universe, [-90, -89, cntrauxdError[0], cntrauxdError[1]])
#dErrorPitch['NM'] = fuzz.trimf(dErrorPitch.universe, [cntrauxdError[0], cntrauxdError[1], cntrauxdError[2]])
#dErrorPitch['NS'] = fuzz.trimf(dErrorPitch.universe, [cntrauxdError[1], cntrauxdError[2], cntrauxdError[3]])
#dErrorPitch['AZ'] = fuzz.trimf(dErrorPitch.universe, [cntrauxdError[2], cntrauxdError[3], cntrauxdError[4]])
#dErrorPitch['PS'] = fuzz.trimf(dErrorPitch.universe, [cntrauxdError[3], cntrauxdError[4], cntrauxdError[5]])
#dErrorPitch['PM'] = fuzz.trimf(dErrorPitch.universe, [cntrauxdError[4], cntrauxdError[5], cntrauxdError[6]])
#dErrorPitch['PL'] = fuzz.trapmf(dErrorPitch.universe, [cntrauxdError[5], cntrauxdError[6], 89,90])






Error.view()
dError.view()
ElevStick.view()
plt.show()


"""Plot Surface"""
x_sampled = np.arange(0, 1.001, 0.01)
y_sampled = np.arange(0, 1.001, 0.01)

x,y = np.meshgrid(x_sampled,y_sampled)
z = np.empty_like(x)

for j, yi in enumerate(y_sampled):
    for i, xi in enumerate(x_sampled):
        sim.input['Error'] = xi
        sim.input['Error Derivative'] = yi
        sim.compute()
        z[j,i] = sim.output['Elevation Stick']

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel("erro (graus)")
ax.set_ylabel("derivada do erro (graus/segundo)")
ax.set_zlabel("Elev Stick")
surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis', linewidth=0.1)
plt.show()