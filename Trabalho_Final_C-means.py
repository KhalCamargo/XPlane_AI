import numpy as np
import skfuzzy as fuzz
from matplotlib import pyplot as plt
import pandas as pd
import skfuzzy.control as ctrl

xlsErro = pd.read_excel('Dados USAR SO PRA PITCH 3.xlsx', sheet_name='Planilha1', usecols = "X,AB", skiprows=160)
xlsdErro = pd.read_excel('Dados USAR SO PRA PITCH 3.xlsx', sheet_name='Planilha1', usecols = "Z,AB", skiprows=160)

xlsErro = xlsErro.to_numpy()
xlsdErro = xlsdErro.to_numpy()

erro = xlsErro[:,0]

cmd = xlsErro[:,1]

derro = xlsdErro[:,0]
cmddErro = xlsdErro[:,1]
print(erro)

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

cluster_membership = np.argmax(uError, axis=0)

fig3, ax3 = plt.subplots()
ax3.set_title('Trained Model')
for j in range(n_clusters):
    ax3.plot(data[cluster_membership == j, 0],
             data[cluster_membership == j, 1], 'o',
             label='series ' + str(j))

for pt in cntrError:
    ax3.plot(pt[0], pt[1], 'rs')

ax3.legend()
plt.show()

""" Controlador:

NS-negative small
NM-negative medium
NL-negative large
AZ-approximately zero
PS-positive small
PM-positive medium
PL-positive large
"""

# Create the three fuzzy variables - two inputs, one output
ErrorPitch = ctrl.Antecedent(np.arange(-16, 16, 0.001), 'Error - Pitch')
dErrorPitch = ctrl.Antecedent(np.arange(-5, 5, 0.001), 'Derivative Error - Pitch')
ElevStick = ctrl.Consequent(np.arange(-0.3, 0.3, 0.001), 'Elevation Stick')

namesStick = ['NL', 'NM', 'NS', 'AZ', 'PS', 'PM', 'PL']
# Auto-membership function population is possible with .automf(3, 5, or 7)

ElevStick.automf(names=namesStick)

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

# Add Rules
rule1 = ctrl.Rule((ErrorPitch['NL'] & dErrorPitch['PL'])|
                  (ErrorPitch['NM'] & dErrorPitch['PM'])|
                  (ErrorPitch['NS'] & dErrorPitch['PS'])|
                  (ErrorPitch['AZ'] & dErrorPitch['AZ'])|
                  (ErrorPitch['PS'] & dErrorPitch['NS'])|
                  (ErrorPitch['PM'] & dErrorPitch['NM'])|
                  (ErrorPitch['PL'] & dErrorPitch['NL']), ElevStick['AZ'])

rule2 = ctrl.Rule((ErrorPitch['PL'] & dErrorPitch['PL'])|
                  (ErrorPitch['PL'] & dErrorPitch['PM'])|
                  (ErrorPitch['PL'] & dErrorPitch['PS'])|
                  (ErrorPitch['PL'] & dErrorPitch['AZ'])|
                  (ErrorPitch['PM'] & dErrorPitch['PL']), ElevStick['NL'])

rule3 = ctrl.Rule((ErrorPitch['NL'] & dErrorPitch['NL'])|
                  (ErrorPitch['NL'] & dErrorPitch['NM'])|
                  (ErrorPitch['NL'] & dErrorPitch['NS'])|
                  (ErrorPitch['NL'] & dErrorPitch['AZ'])|
                  (ErrorPitch['NM'] & dErrorPitch['NL']), ElevStick['PL'])

rule4 = ctrl.Rule((ErrorPitch['NS'] & dErrorPitch['AZ'])|
                  (ErrorPitch['NM'] & dErrorPitch['NS'])|
                  (ErrorPitch['NL'] & dErrorPitch['NM'])|
                  (ErrorPitch['AZ'] & dErrorPitch['PS'])|
                  (ErrorPitch['NM'] & dErrorPitch['NL']), ElevStick['PS'])

rule5 = ctrl.Rule(ErrorPitch['NM'] | dErrorPitch['NM'], ElevStick['PM'])
rule6 = ctrl.Rule(ErrorPitch['NS'] | dErrorPitch['NS'], ElevStick['PS'])
rule7 = ctrl.Rule(ErrorPitch['AZ'] | dErrorPitch['AZ'], ElevStick['AZ'])
rule8 = ctrl.Rule(ErrorPitch['PM'] | dErrorPitch['PM'], ElevStick['NS'])
rule9 = ctrl.Rule(ErrorPitch['PL'] | dErrorPitch['PS'], ElevStick['PS'])

rule1.view()

"""Controller"""

#Add controller
System = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5,rule6, rule7, rule8])
sim = ctrl.ControlSystemSimulation(System)

"""
sim.input['Error - Pitch'] = 3
sim.input['Derivative Error - Pitch'] = 1
sim.compute()

print(sim.output['Elevation Stick'])
ElevStick.view(sim=sim)
"""
"""Plot Surface"""

x_sampled = np.arange(-15, 15, 0.1)
y_sampled = np.arange(-15, 15, 0.1)

x,y = np.meshgrid(x_sampled,y_sampled)
z = np.zeros_like(x)

for i, xi in enumerate(x_sampled):
    for j, yi in enumerate(y_sampled):
        sim.input['Error - Pitch'] = xi
        sim.input['Derivative Error - Pitch'] = yi
        sim.compute()
        z[i,j] = sim.output['Elevation Stick']

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel("erro (graus)")
ax.set_ylabel("derivada do erro (graus/segundo)")
ax.set_zlabel("Elev Stick")
surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis', linewidth=0.1)
plt.show()