import os 
import numpy as np 
import skfuzzy as fuzz 
from skfuzzy import control as ctrl 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

# parcialmente baseado no exemplo em https://pythonhosted.org/scikit-fuzzy/auto_examples/plot_control_system_advanced.html 

# INVERTED PENDULUM 

angle_universe = np.arange(-45, 45, 1) # degrees
output_universe = np.arange(-10, 10, 0.1) # volts

error = ctrl.Antecedent(angle_universe, 'erro') 
errord = ctrl.Antecedent(angle_universe, 'derivada do erro')
output = ctrl.Consequent(output_universe, 'saida')

linguistic_values_errord = ['NP', 'AZ', 'PP']
linguistic_values_error = ['NM', 'NP', 'AZ', 'PP', 'PM'] 
linguistic_values_output = linguistic_values_error 

# crior outomoticamente as funcoes de pertencimento 
error.automf(names=linguistic_values_error) 
errord.automf(names=linguistic_values_errord) 
output.automf(names=linguistic_values_output) 

rule0 = ctrl.Rule(antecedent=((error['NM'] & errord['AZ'])), consequent=output['NM'], label='regra 0')
rule1 = ctrl.Rule(antecedent=((error['NP'] & errord['NP'])), consequent=output['NP'], label='regra 1') 
rule2 = ctrl.Rule(antecedent=((error['NP'] & errord['PP'])), consequent=output['AZ'], label='regra 2') 
rule3 = ctrl.Rule(antecedent=((error['AZ'] & errord['AZ'])), consequent=output['AZ'], label='regra 3')
rule4 = ctrl.Rule(antecedent=((error['PP'] & errord['NP'])), consequent=output['AZ'], label='regra 4')
rule5 = ctrl.Rule(antecedent=((error['PP'] & errord['PP'])), consequent=output['PP'], label='regra 5')
rule6 = ctrl.Rule(antecedent=((error['PM'] & errord['AZ'])), consequent=output['PM'], label='regra 6') 

# criacao do controlador
system = ctrl.ControlSystem(rules=[rule0, rule1, rule2, rule3, rule4, rule5, rule6])

# criacao da simulacao do controlador
sim = ctrl.ControlSystemSimulation(system)

# exemplo de entrada e verificacdo da saida
# sim.input['erro'] = -44
# sim.input['derivada do erro'] = 0 
# sim.compute()

# mostra o erro
# error.view()
# errord.view()

# mostra a saida
# print(sim.output['saida'])
# output.view(sim=sim)
# plt.show()

# fazer loop pars nostror todas as saidas 
x_sampled = np.arange(-44, 44, 5) 
y_sampled = np.arange(-44, 44, 5) 

x,y = np.meshgrid(x_sampled,y_sampled)
z = np.zeros_like(x) 

for i, xi in enumerate(x_sampled): 
    for j, yi in enumerate(y_sampled): 
        sim.input['erro'] = xi 
        sim.input['derivada do erro'] = yi
        sim.compute()
        z[i,j] = sim.output['saida'] 
        
# plot
fig = plt.figure() 
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel("erro (graus)") 
ax.set_ylabel("derivada do erro (graus/segundo)") 
ax.set_zlabel("tensao eletrica (V)") 
surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis', linewidth=0.1)
plt.show()