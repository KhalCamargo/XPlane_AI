import numpy as np
import math
from sys import getsizeof
from datetime import timedelta
from typing import List
from datetime import datetime
from struct import *
import socket
import pandas as pd
import xlrd as xlrd
import skfuzzy as fuzz
from matplotlib import pyplot as plt
import skfuzzy.control as ctrl
import keyboard as k
import tkinter as tk
from simple_pid import PID

# Definição das funções de pertinencia
def trimf (x, a , b , c ):
   
    y = list()
    if type(x) is np.ndarray: x = list(x)
    elif type(x) is not list: x = [x]
    else: x = list(x)
    for el in x:
        val = max(min((el-a)/(b-a),(c-el)/(c-b)),0.0)
        y.append(val)
    return np.array(y)

def trapmf(x,a,b,c,d):
    y = list()
    if type(x) is np.ndarray: x = list(x)
    elif type(x) is not list: x = [x]
    else: x = list(x)
    for el in x:
        val = max(min((el-a)/(b-a),1.0,(d-el)/(d-c)),0.0)
        y.append(val)
    return np.array(y)

def sigmf(x,down,top):
    w = 1/((top - down)/(x.max()-x.min()))
    c = np.minimum(top,down) + abs((top-down)/2)
    y = list()
    if type(x) is np.ndarray: x = list(x)
    elif type(x) is not list: x = [x]
    else: x = list(x)
    for el in x:
        try:
            den = 1 + math.exp(-w*(el-c))
        except OverflowError:
            den = float('inf')
        val = 1/(den)
        y.append(val)
    return np.array(y)

def gauss2mf(x,a,b,c):
    s1 = float(abs(a-b))/3
    s2 = float(abs(b-c))/3
    y = list()
    if type(x) is np.ndarray: x = list(x)
    elif type(x) is not list: x = [x]
    else: x = list(x)
    for el in x:
        if el <= b:
            val = math.exp(-(el-b)*2/(2*s1*2))
        else:
            val = math.exp(-(el-b)*2/(2*s2*2))
        y.append(val)
    return np.array(y)


# Definição da função de Defuzzyficação
def defuzzCentroid(x,y):
        x = np.array(x)
        y = np.array(y)
        num = np.sum(x*y)
        den = np.sum(y)
        if den == 0 and num == 0:            
            return 0;
        out = num/den
        return out.tolist()

# Definição da biblioteca Fuzzy desenvolvida pelos autores
class CDataFuzz:
    def _init_(self,ref,val):
        self._ref = ref
        self._val = val
    def val(self):
        return self._val
    def ref(self):
        return self._ref
    def set(self,ref,val):
        self._ref = ref
        self._val = val
    def _call_(self,input):
        return self._val(input)

class CGauss2MemberFunction:
    def _init_(self, name : str, Lims : List[float]):
        self._name = name
        self._low = Lims[0]
        self._med = Lims[1]
        self._high = Lims[2]
    def _call_(self,x):
        return gauss2mf(x,self._low,self._med,self._high)
    def name(self):
        return self._name
    def lims(self):
        return [self._low,self._med,self._high]

class CSigMemberFunction:
    def _init_(self, name : str, Lims : List[float]):
        self._name = name
        self._low = Lims[0]        
        self._high = Lims[1]
    def _call_(self,x):
        return sigmf(x,self._low,self._high)
    def name(self):
        return self._name
    def lims(self):
        return [self._low,self._high]

class CTriMemberFunction:
    def _init_(self, name : str, triLims : List[float]):
        self._name = name
        self._low = triLims[0]
        self._med = triLims[1]
        self._high = triLims[2]
    def _call_(self,x):
        return trimf(x,self._low,self._med,self._high)
    def name(self):
        return self._name
    def lims(self):
        return [self._low,self._med,self._high]

class CTrapMemberFunction:
    def _init_(self, name : str, trapLims : List[float]):
        self._name = name
        self._low = trapLims[0]
        self._medLow = trapLims[1]
        self._medHigh = trapLims[2]
        self._high = trapLims[3]
    def _call_(self,x):
        return trapmf(x,self._low,self._medLow,self._medHigh,self._high)
    def name(self):
        return self._name

class CFuzzyVar:
    def _init_(self,funList : List[CTriMemberFunction],limits:List[float],name:str):
        self._memberFunctions = funList
        self._name = name
        self._limits = limits
    def _call_(self,x):
        y = list()
        for fn in self._memberFunctions:
            y.append(fn(x))
        return y
    def name(self):
        return self._name
    def lims(self):
        return self._limits
    def _getitem_(self,key):
        for fn in self._memberFunctions:
            if key == fn.name():
                return CDataFuzz(self,fn)
        return CDataFuzz(self,self._memberFunctions.end())


# Variavel de controle para saber se estamos rodando a primeira vez
firstRun = False

#Método para coletar dados
def getData():
    
    if firstRun:

        dataR = pd.read_excel('C:\\Workspace\\XPlane\\Dados Roll e Pitch 953.xlsx')
        dataR = dataR.drop(dataR.index[0:953]) #Removing n-first rows

        data = dataR #using new data

        data = data.loc[(abs(data['   pitch,__deg '])) <= 15] #Selecting only pitch within range
        data = data.loc[:,'ErroPitch':]

        
        datas = data.loc[:,['ErroPitch','dErroP','dCmdPitch']]
        

        datas.loc[:,'dErroP'] = datas.loc[:,'dErroP'].mul(-1)

        datasR = datas
        datas.to_pickle('datasetP.pkl')
        datasR.to_pickle('datasetR.pkl')
    else:
        datas = pd.read_pickle('datasetP.pkl')
        datasR = pd.read_pickle('datasetR.pkl')

    #datas.loc[:,'ErroPitch':'dErroP'] = (datas.loc[:,'ErroPitch':'dErroP'] - datas.loc[:,'ErroPitch':'dErroP'].min())/(datas.loc[:,'ErroPitch':'dErroP'].max()-datas.loc[:,'ErroPitch':'dErroP'].min())
    return datas, datasR

#Método para normalizar dados
def normalizeDatas(datas):
    l_datas = datas.copy()
    l_datas.iloc[:,:-1] = (l_datas.iloc[:,:-1] - l_datas.iloc[:,:-1].min())/(l_datas.iloc[:,:-1].max()-l_datas.iloc[:,:-1].min())
    return l_datas
#Método para normalizar novos dados vindouros
def normalizeNewData(data,datas):
    colsName = datas.columns.values.tolist()
    datas = datas.iloc[:,:-1]
    toClassDF = {colsName[0]:[data[0][0]],
                 colsName[1]:[data[0][1]]}
    toClass = pd.DataFrame(toClassDF,columns=colsName[:-1])
    toClass = (toClass -  datas.min())/(datas.max()-datas.min())
    toClass[colsName[0]].loc[(toClass[colsName[0]] > 1)] = 1
    toClass[colsName[0]].loc[(toClass[colsName[0]] < 0)] = 0
    toClass[colsName[1]].loc[(toClass[colsName[1]] > 1)] = 1
    toClass[colsName[1]].loc[(toClass[colsName[1]] < 0)] = 0
    return toClass.to_numpy()
#Método para gerar sistema Fuzzy
def generateAutoFuzzy(datas):
    #Calcula os clusters
    datasnp = datas.to_numpy()
    n_clusters = 7
    cntrError, uError, u0Error, dError, jmError, pError, fpcError = fuzz.cluster.cmeans(datasnp.T, c=n_clusters, m=2, error=0.005, maxiter=1000, init=None)
    #Ordena a coordenada Z dos clusters
    cntrStick = []
    for i in range (0,n_clusters):
        cntrStick.append(cntrError[i,2])
    cntrStick.sort()
    # Cria as funções de pertinencia da saída (Z)
    mfList = list()
    for i in range(0,n_clusters):
        if i == 0:
            mf = CSigMemberFunction(str(i),[cntrStick[i+1],cntrStick[i]])
        elif i == (n_clusters-1):
            mf = CSigMemberFunction(str(i),[cntrStick[i-1],cntrStick[i]])
            toto =3
        else:
            mf = CGauss2MemberFunction(str(i),[cntrStick[i-1],cntrStick[i],cntrStick[i+1]])
        mfList.append(mf)

        #Plots para debug
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111, projection='3d')

    ax3.set_title('Trained Model')

    cluster_membership = np.argmax(uError, axis=0)
    for j in range(n_clusters):
        ax3.scatter(datasnp[cluster_membership == j, 0],
                 datasnp[cluster_membership == j, 1],datasnp[cluster_membership == j, 2], 'o',
                 label='series ' + str(j),alpha=0.5)


    for pt in cntrError:
        ax3.scatter(pt[0], pt[1],pt[2], marker='s',c='r',s=15**2)
    plt.xlabel('Erro Normalizado')
    plt.ylabel('Derivada Normalizada do Erro')
    plt.title('Dados coletados para variação de Stick e centro dos Clusters')
    plt.legend(loc='lower right')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    x = np.arange(-1,1.0001,0.0001)
    i = 0
    for f in mfList:
        ax.plot(x,f(x))
    plt.title('Funções de Pertinência para a Saída')
    ax.set_ylabel('Pertinência')
    ax.set_xlabel('Universo da saída')
    ax.set_xlim([-0.025,0.025])
    plt.show()

    #Saída é lista com as funções de pertinencia
    Out = CFuzzyVar(mfList,[-1,1],'Stick')

    return Out,cntrError #retorna as funções de pertinencia e os Clusters

#Calcula a saída de um sistema fuzzy com base nos cluster e novos dados normalizados
def outCalc(outVar : CFuzzyVar, centroids, newNormData):

    #ordena os centroides pela coord Z
    centroids = centroids[centroids[:,2].argsort()]

    #realiza a predict com base na projecao nas dimensoes de entrada 
    u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
    newNormData.T, centroids[:,0:2], 2, error=0.005, maxiter=1000)

    x = np.arange(-1,1.001,0.001)
    #para cada cluster, com base na pertinencia, calcula a geometria da saída
    finalY = np.zeros_like(x)
    for i, pert in enumerate(u):
       mf = outVar[str(i)].val()
       pertLim = np.full_like(x,pert)
       y = mf(x)
       y = np.minimum(y,pertLim)
       finalY = np.maximum(finalY,y) #Compoe a saída via MAX

    defuzzedOut = defuzzCentroid(x.tolist(),finalY.tolist()) #Defuzzyfica a saída
    return defuzzedOut
       

## XPLANE FUZZY CONTROL ##

# Creating Socket
UDP_IP = "127.0.0.1"
UDP_PORT = 49000
recSock = socket.socket(socket.AF_INET,  socket.SOCK_DGRAM)
sendSock = socket.socket(socket.AF_INET,  socket.SOCK_DGRAM)


# Dataref subscribe
AGLDRef = "sim/cockpit2/gauges/indicators/pitch_AHARS_deg_pilot"
AGLDRefPadded = AGLDRef.ljust(400,"\0")
AGLDRefPadded = bytes(AGLDRefPadded,'utf-8')
RREFstr = 'RREF\0'
RREFstr = bytes(RREFstr,'utf-8')
requestAGL = RREFstr + pack('ii',2,1) + AGLDRefPadded

PRate = "sim/flightmodel/position/Q"
PRatePadded = PRate.ljust(400,"\0")
PRatePadded = bytes(PRatePadded,'utf-8')
requestPRate = RREFstr + pack('ii',2,2) + PRatePadded

Roll = "sim/cockpit2/gauges/indicators/roll_AHARS_deg_pilot"
RollPadded = Roll.ljust(400,"\0")
RollPadded = bytes(RollPadded,'utf-8')
requestRoll = RREFstr + pack('ii',2,3) + RollPadded

RRate = "sim/flightmodel/position/P"
RRatePadded = RRate.ljust(400,"\0")
RRatePadded = bytes(RRatePadded,'utf-8')
requestRRate = RREFstr + pack('ii',2,4) + RRatePadded


# Set pitch command
yawDREF = "sim/cockpit2/controls/yoke_pitch_ratio"
yawDREFPadded = yawDREF.ljust(500,"\0")
yawDREFPadded = bytes(yawDREFPadded,'utf-8')
DREFstr = 'DREF\0'
DREFstr = bytes(DREFstr,'utf-8')

JoyPPadded = yawDREF.ljust(400,"\0")
JoyPPadded = bytes(yawDREF,'utf-8')
requestJoyP = RREFstr + pack('ii',2,5) + JoyPPadded 

# Set roll command
ailDREF = "sim/cockpit2/controls/yoke_roll_ratio"
ailDREFPadded = ailDREF.ljust(500,"\0")
ailDREFPadded = bytes(ailDREFPadded,'utf-8')


#método para receber dados
def receiveData():
    data = recSock.recv(1024)
    data = data[5:]
    test = len(data)
    id1, altitude, id2, PRate, id3, roll, id4, RRate = unpack('ifififif',data)
    if id1 == 1 and id2 == 2 and id3 == 3 and id4 == 4:
        out = [altitude,PRate, roll, RRate]
    
    return out

#método para enviar comandos ao xplane
def sendCommand(commandP, commandR):

    commandRoll = DREFstr + pack('f',commandR) + ailDREFPadded
    sendSock.sendto(commandRoll,(UDP_IP,UDP_PORT))

    commandYaw = DREFstr + pack('f',commandP) + yawDREFPadded
    sendSock.sendto(commandYaw,(UDP_IP,UDP_PORT))
    

#se inscreve nos datarefs
recSock.sendto(requestAGL,(UDP_IP,UDP_PORT))
recSock.sendto(requestPRate,(UDP_IP,UDP_PORT))
recSock.sendto(requestRoll,(UDP_IP,UDP_PORT))
recSock.sendto(requestRRate,(UDP_IP,UDP_PORT))

#variaveis de estado
RUN = True
FUZZY = True

#metodos para GUI
def setController():
    FUZZY = ButVar.get()

master = tk.Tk()
w1 = tk.Scale(master, from_=40, to=-40, tickinterval = 2)
w1.set(0)
w1.pack()

ButVar = tk.BooleanVar()
Button1 = tk.Radiobutton(master, text = 'Fuzzy', variable = ButVar, value = True, command = setController)
Button1.pack(anchor=tk.W)
Button2 = tk.Radiobutton(master, text = 'Proportional', variable = ButVar, value = False, command = setController)
Button2.pack(anchor=tk.W)

Button1.select()
Button2.deselect()

#Referencia conforme valor na GUI
C_REF = w1.get()
FUZZY = False
USE_PID = True
USE_BANGBANG = False

# Prepares PID controller (check these values -> what is the setpoint?)
pid_pitch = PID( Kp = 0.0136, Ki = 0.0148, Kd = 0.004, setpoint = 20, sample_time = 0.01, output_limits = (None, None))
pid_roll = PID( Kp = 0.0136, Ki = 0.0148, Kd = 0.004, setpoint = 20, sample_time = 0.01, output_limits = (None, None))

#Pega os dados para Pitch e Roll
datas, datasR = getData()

#Normaliza os dados
norm_datas = normalizeDatas(datas)
norm_datasR = normalizeDatas(datasR)

#Gera sistemas Fuzzy
OutVar, centroids = generateAutoFuzzy(norm_datas)
OutVarR, centroidsR = generateAutoFuzzy(norm_datasR)

#Variáveis para armazenamento do dado anterior

oldcmd = 0
oldmdr = 0
#para cada ciclo
while (true):
    if run: #se está rodando
        #recebe dados
        datarec = receivedata()

        #define os erros e derivadas    
        errop = datarec[0] - c_ref
        derrop = datarec[1]

        error = datarec[2]
        derror = datarec[3]

        if fuzzy: #se fuzzy

            #normaliza e calcula as saidas
            norm_inp = normalizenewdata([[errop,derrop]],datas)
            norm_inr = normalizenewdata([[error,derror]],datasr)
            commandp = outcalc(outvar, centroids, norm_inp)
            commandr = outcalc(outvarr, centroidsr, norm_inr)

            #incrementa comando conforme saída
            commandr = commandr*0.5 + oldcmdr
            commandp = commandp + oldcmd

            #evita saturação
            if commandp > 1:
                commandp = 1
            elif commandp < -1:
                commandp = -1

            if commandr > 1:
                commandr = 1
            elif commandr < -1:
                commandr = -1
        if USE_PID:
            output_pitch = pid_pitch(current_value_pitch) # de onde vem o current value? é o errop ou o derrop?
            output_roll = pid_roll(current_value_roll) # de onde vem o current value? é o error ou o derror?
            commandp = output_pitch
            commandr = output_roll
        if USE_BANGBANG:
            if current_value_pitch >  0:
                commandp = -1
            else if current_value_pitch < 0:
                commandp = 1
            
            if current_value_roll > 0:
                commandr = -1
            else if current_value_roll < 0:
                commandp = 1
        else: #se nao fuzzy, proporcional
            k = 1
            nerro= normalizenewdata([[errop,derrop]],datas)
            commandp = -k*(nerro[0][0]*2 - 1)
            #roll ainda é fuzzy
            norm_inr = normalizenewdata([[error,derror]],datasr)
            commandr = outcalc(outvarr, centroidsr, norm_inr)
            commandr = commandr*0.5 + oldcmdr
            #evita fuzzy
            if commandr > 1:
                commandr = 1
            elif commandr < -1:
                commandr = -1

        print(f'error | derror | command ||| ||| {errop} | {derrop} | {commandp} ')
        sendcommand(commandp,commandr) #envia os comandos
    try:  # used try so that if user pressed other than the given key error will not be shown
        if k.is_pressed('s'):  # if key 's' is pressed 
            if run:
                #se rodando, para o controle e cancela recebimento dos dados
                requestprate = rrefstr + pack('ii',0,2) + pratepadded
                requestagl = rrefstr + pack('ii',0,1) + agldrefpadded
                requestroll = rrefstr + pack('ii',0,3) + rollpadded
                requestrrate = rrefstr + pack('ii',0,4) + rratepadded

                recsock.sendto(requestagl,(udp_ip,udp_port))
                recsock.sendto(requestprate,(udp_ip,udp_port))
                recsock.sendto(requestroll,(udp_ip,udp_port))
                recsock.sendto(requestrrate,(udp_ip,udp_port))
                run = false
                print("stopped")
        elif k.is_pressed('r'):
            if run == false:
                #se parado, roda e pede dados novamente ao xplane
                requestprate = rrefstr + pack('ii',2,2) + pratepadded
                requestagl = rrefstr + pack('ii',2,1) + agldrefpadded
                requestroll = rrefstr + pack('ii',2,3) + rollpadded
                requestrrate = rrefstr + pack('ii',2,4) + rratepadded

                recsock.sendto(requestagl,(udp_ip,udp_port))
                recsock.sendto(requestprate,(udp_ip,udp_port))
                recsock.sendto(requestroll,(udp_ip,udp_port))
                recsock.sendto(requestrrate,(udp_ip,udp_port))
                run = true
                print("running")
        elif k.is_pressed('f'):
            #alterna entre os controladores
            fuzzy = not fuzzy
            print(f'using fuzzy? {fuzzy}')
    except:
        pass
    #atualiza referencia, valores antigos e GUI
    c_ref = w1.get()
    oldcmd = commandp
    oldcmdr = commandr
    master.update_idletasks()
    master.update()



#Plot da superficie

err = np.arange(0,1.1,0.1)
derr = err
z = list()
zR = list()
for elerr in err:
    for elderr in derr:
        result = outCalc(OutVar,centroids,np.array([[elerr,elderr]]))
        resultR = outCalc(OutVarR,centroidsR,np.array([[elerr,elderr]]))
        z.append(result)
        zR.append(resultR)

X,Y = np.meshgrid(err,derr)
Z = np.reshape(z,(int(1.1/0.1),int(1.1/0.1)))
ZR = np.reshape(zR,(int(1.1/0.1),int(1.1/0.1)))

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.plot_surface(X, Y, Z,rstride=1,cstride=1,cmap='viridis',edgecolor='none')
plt.xlabel('Erro Normalizado')
plt.ylabel('Derivada Normalizada do Erro')
plt.title('Variação a ser aplicada no Stick')

#figR = plt.figure()
#axR = figR.add_subplot(111,projection='3d')
#axR.plot_surface(X, Y, ZR,rstride=1,cstride=1,cmap='viridis',edgecolor='none')

plt.show()