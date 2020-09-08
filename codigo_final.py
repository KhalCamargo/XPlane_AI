import numpy as np
import math
from sys import getsizeof
from datetime import timedelta
from typing import List
from datetime import datetime
from struct import *
import socket
import pandas as pd
#import xlrd as xlrd
import skfuzzy as fuzz
from matplotlib import pyplot as plt
import skfuzzy.control as ctrl
import keyboard as k
import tkinter as tk
import keras as ker
from keras.models import Sequential
from keras.layers import Dense
from simple_pid import PID

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

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
    w = (1/(top - down))*10
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

##Método para coletar dados


#Método para coletar dados
def getData():
    
    if firstRun:

        data1 = pd.read_excel('C:\\Workspace\\XPlane\\Final_Pitch_Beach_750.xlsx')
        data1 = data1.drop(data1.index[0:750]) #Removing n-first rows
        data1 = data1.loc[(abs(data1['   pitch,__deg '])) <= 10] #Selecting only pitch within range

        data2 = pd.read_excel('C:\\Workspace\\XPlane\\Final_Pitch_Beach_930.xlsx')
        data2 = data2.drop(data2.index[0:930]) #Removing n-first rows
        data2 = data2.loc[(abs(data2['   pitch,__deg '])) <= 10] #Selecting only pitch within range

        data3 = pd.read_excel('C:\\Workspace\\XPlane\\Final_Pitch_Beach_1094.xlsx')
        data3 = data3.drop(data3.index[0:1094]) #Removing n-first rows
        data3 = data3.loc[(abs(data3['   pitch,__deg '])) <= 10] #Selecting only pitch within range


        dataR = data1.append(data2)

        dataR = dataR.append(data3)
        
        data = dataR #using new data

        
        data = data.loc[:,'ErroPitch':]

        
        datas = data.loc[:,['ErroPitch','dErroP','CmdPitch']]
        

        datas.loc[:,'dErroP'] = datas.loc[:,'dErroP'].mul(-1)

        datasR = datas
        datas.to_pickle('datasetPNew.pkl')
        datasR.to_pickle('datasetRNew.pkl')
    else:
        datas = pd.read_pickle('datasetPNew.pkl')
        datasR = pd.read_pickle('datasetRNew.pkl')

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
        cntrStick.append(cntrError[i,-1]) #mudar 2 para -1
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
    #fig3 = plt.figure()
    #ax3 = fig3.add_subplot(111, projection='3d')

    #ax3.set_title('Trained Model')

    #cluster_membership = np.argmax(uError, axis=0)
    #for j in range(n_clusters):
    #    ax3.scatter(datasnp[cluster_membership == j, 0],
    #             datasnp[cluster_membership == j, 1],datasnp[cluster_membership == j, -1], 'o',
    #             label='series ' + str(j),alpha=0.05)


    #for pt in cntrError:
    #    ax3.scatter(pt[0], pt[1],pt[2], marker='s',c='r',s=15**2)
    #plt.xlabel('Erro Normalizado')
    #plt.ylabel('Derivada Normalizada do Erro')
    #plt.title('Dados coletados para variação de Stick e centro dos Clusters')
    #plt.legend(loc='upper right')
    #plt.show()

    #fig = plt.figure()
    #ax = fig.add_subplot(1, 1, 1)
    #x = np.arange(-1,1.0001,0.0001)
    #i = 0
    #for f in mfList:
    #    ax.plot(x,f(x))
    #plt.title('Funções de Pertinência para a Saída')
    #ax.set_ylabel('Pertinência')
    #ax.set_xlabel('Universo da saída')
    ##ax.set_xlim([-0.025,0.025])
    #plt.show()

    #Saída é lista com as funções de pertinencia
    Out = CFuzzyVar(mfList,[-1,1],'Stick')

    return Out,cntrError #retorna as funções de pertinencia e os Clusters

#Calcula a saída de um sistema fuzzy com base nos cluster e novos dados normalizados
def outCalc(outVar : CFuzzyVar, centroids, newNormData):

    #ordena os centroides pela coord Z
    centroids = centroids[centroids[:,-1].argsort()] #mudar 2 para -1

    #realiza a predict com base na projecao nas dimensoes de entrada 
    u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
    newNormData.T, centroids[:,0:-1], 2, error=0.005, maxiter=1000) #mudar 2 para -1

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

def DataRefByteGen(name,id,freq=2):
    DRef = name
    DRefPadded = DRef.ljust(400,"\0")
    DRefPadded = bytes(DRefPadded,'utf-8')
    RREFstr = 'RREF\0'
    RREFstr = bytes(RREFstr,'utf-8')
    request = RREFstr + pack('ii',int(freq),int(id)) + DRefPadded
    return request


requestAGL = DataRefByteGen("sim/cockpit2/gauges/indicators/pitch_AHARS_deg_pilot",1)

requestPRate = DataRefByteGen("sim/flightmodel/position/Q",2)

requestRoll = DataRefByteGen("sim/cockpit2/gauges/indicators/roll_AHARS_deg_pilot",3)

requestRRate = DataRefByteGen("sim/flightmodel/position/P",4)

requestJoyP = DataRefByteGen("sim/cockpit2/controls/yoke_pitch_ratio",5)

requestSpeed = DataRefByteGen("sim/cockpit2/gauges/indicators/airspeed_kts_pilot",6)

requestDensityRation = DataRefByteGen("sim/weather/sigma",7)

requestLiftForce = DataRefByteGen("sim/flightmodel/forces/fnrml_aero",8) #In Newtons!

# Set pitch command
def WriteToDrefByteGen(name):
    DREF = name
    DREFPadded = DREF.ljust(500,"\0")
    DREFPadded = bytes(DREFPadded,'utf-8')
    DREFstr = 'DREF\0'
    DREFstr = bytes(DREFstr,'utf-8')
    return DREFstr, DREFPadded


DREFstr, yawDREFPadded = WriteToDrefByteGen("sim/cockpit2/controls/yoke_pitch_ratio")


# Set roll command

DREFstr, ailDREFPadded = WriteToDrefByteGen("sim/cockpit2/controls/yoke_roll_ratio")


#método para receber dados
def receiveData():
    data = recSock.recv(1024)
    data = data[5:]
    test = len(data)
    id1, altitude, id2, PRate, id3, roll, id4, RRate, id5, PitchCmd, id6, Speed , id7, densityRatio , id8, LiftN= unpack('ifififififififif',data)
    if id1 == 1 and id2 == 2 and id3 == 3 and id4 == 4:
        out = [altitude,PRate, roll, RRate, PitchCmd,Speed,densityRatio,(LiftN*0.225)]
    
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

recSock.sendto(requestJoyP,(UDP_IP,UDP_PORT))
recSock.sendto(requestSpeed,(UDP_IP,UDP_PORT))
recSock.sendto(requestDensityRation,(UDP_IP,UDP_PORT))
recSock.sendto(requestLiftForce,(UDP_IP,UDP_PORT))

#variaveis de estado
RUN = True
FUZZY = True

#metodos para GUI
def setController():
    FUZZY = ButVar

master = tk.Tk()
w1 = tk.Scale(master, from_=40, to=-40, tickinterval = 2)
w1.set(0)
w1.pack()

w2 = tk.Scale(master, from_=0, to=4, tickinterval = 1)
w2.set(0)
w2.pack()

w3 = tk.Scale(master, from_=0, to=1, resolution = 0.0001, tickinterval = 0.0001)
w3.set(0.0336)
w3.pack()

w4 = tk.Scale(master, from_=0, to=2, resolution = 0.0001,tickinterval = 0.0001)
w4.set(2)
w4.pack()

w5 = tk.Scale(master, from_=0, to=1, resolution = 0.0001, tickinterval = 0.0001)
w5.set(0.0)
w5.pack()


#Referencia conforme valor na GUI
C_REF = w1.get()



controller = w2.get()

def getController(val):
    return {
        0: 'Fuzzy',
        1: 'Prop',
        2: 'Bang',
        3: 'PID',
        4: 'Neuro',
        }[val]



#Pega os dados para Pitch e Roll
datas, datasR = getData()

#Normaliza os dados
norm_datas = normalizeDatas(datas)
norm_datasR = normalizeDatas(datasR)

#Gera sistemas Fuzzy
OutVar, centroids = generateAutoFuzzy(norm_datas)
OutVarR, centroidsR = generateAutoFuzzy(norm_datasR)

#Sistema PID
pid_pitch = PID( Kp = w3.get(), Ki = w4.get(), Kd = w5.get(), setpoint = C_REF, sample_time = None, output_limits = (-1, 1))

#Sistema Neuro

TRAIN = False

model = Sequential()

if TRAIN:
    #Gera datasets
    X = norm_datas.loc[:,['ErroPitch','dErroP']]
    y = norm_datas.loc[:,'CmdPitch']

    # Gera modelo

    model = Sequential()
    model.add(Dense(2, input_dim = 2, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='linear'))

    # Compila modelo
    model.compile(loss='mse', optimizer='sgd', metrics=['mse'])

    # Fit the model
    history = model.fit(X, y, epochs=100, batch_size=4,validation_split = 0.2)

    hist_df = pd.DataFrame(history.history)

    hist_csv_file = 'history.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

    model.save('XModel.h5')
else:
    model = ker.models.load_model('XModel_88_epc16_val04_batch1.h5')

#Variáveis para armazenamento do dado anterior

oldcmd = 0
oldcmdr = 0
#para cada ciclo
while (True):
    if RUN: #se está rodando
        #recebe dados
        datarec = receiveData()

        #define os erros e derivadas    
        errop = datarec[0] - C_REF
        derrop = datarec[1]

        error = datarec[2]
        derror = datarec[3]

        #speed = datarec[5]

        #density = datarec[6]

        lift = datarec[7]

        if getController(controller) == 'Fuzzy': #se fuzzy

            #normaliza e calcula as saidas
            norm_inp = normalizeNewData([[errop,derrop]],datas)
            norm_inr = normalizeNewData([[error,derror]],datasR)
            commandp = outCalc(OutVar, centroids, norm_inp)
            commandr = outCalc(OutVarR, centroidsR, norm_inr)

            #incrementa comando conforme saída
            commandr = commandr*0.5 + oldcmdr
            commandp = commandp + oldcmd

            ##calcula a dependencia
            #commandp = commandp/(math.cos(math.radians(error)))

            #evita saturação
            if commandp > 1:
                commandp = 1
            elif commandp < -1:
                commandp = -1

            if commandr > 1:
                commandr = 1
            elif commandr < -1:
                commandr = -1
        elif getController(controller) == 'Prop' : #se nao fuzzy, proporcional
            k = 1
            nerro= normalizeNewData([[errop,derrop]],datas)
            commandp = -k*(nerro[0][0]*2 - 1)
            
            #roll ainda é fuzzy
            norm_inr = normalizeNewData([[error,derror]],datasR)
            commandr = outCalc(OutVarR, centroidsR, norm_inr)
            commandr = commandr*0.5 + oldcmdr


            #evita fuzzy
            if commandr > 1:
                commandr = 1
            elif commandr < -1:
                commandr = -1

        elif getController(controller) == 'Bang':
            if errop >  0.5:
                commandp = -1
            elif errop < 0.5:
                commandp = 1
            else:
                commandp = 0

            #roll ainda é fuzzy
            norm_inr = normalizeNewData([[error,derror]],datasR)
            commandr = outCalc(OutVarR, centroidsR, norm_inr)
            commandr = commandr*0.5 + oldcmdr

            if commandr > 1:
                commandr = 1
            elif commandr < -1:
                commandr = -1

        elif getController(controller) == 'PID':
            output_pitch = pid_pitch(datarec[0])            
            commandp = output_pitch

            #roll ainda é fuzzy
            norm_inr = normalizeNewData([[error,derror]],datasR)
            commandr = outCalc(OutVarR, centroidsR, norm_inr)
            commandr = commandr*0.5 + oldcmdr

            if commandr > 1:
                commandr = 1
            elif commandr < -1:
                commandr = -1

        elif getController(controller) == 'Neuro':
            norm_inp = normalizeNewData([[errop,derrop]],datas)
            output_pitch = model(norm_inp)           
            commandp = output_pitch[0][0]


            #roll ainda é fuzzy
            norm_inr = normalizeNewData([[error,derror]],datasR)
            commandr = outCalc(OutVarR, centroidsR, norm_inr)
            commandr = commandr*0.5 + oldcmdr

            if commandp > 1:
                commandp = 1
            elif commandp < -1:
                commandp = -1

            if commandr > 1:
                commandr = 1
            elif commandr < -1:
                commandr = -1
            

        print(f'error | derror | lift |command ||| ||| {errop} | {derrop} | {lift} | {commandp} ')
        sendCommand(commandp,commandr) #envia os comandos
    try:  # used try so that if user pressed other than the given key error will not be shown
        if k.is_pressed('s'):  # if key 's' is pressed 
            if RUN:
                #se rodando, para o controle e cancela recebimento dos dados
                

                requestagl = DataRefByteGen("sim/cockpit2/gauges/indicators/pitch_AHARS_deg_pilot",1,0)

                
                requestprate = DataRefByteGen("sim/flightmodel/position/Q",2,0)

                requestroll = DataRefByteGen("sim/cockpit2/gauges/indicators/roll_AHARS_deg_pilot",3,0)

                requestrrate = DataRefByteGen("sim/flightmodel/position/P",4,0)

                requestjoyp = DataRefByteGen("sim/cockpit2/controls/yoke_pitch_ratio",5,0)

                requestspeed = DataRefByteGen("sim/cockpit2/gauges/indicators/airspeed_kts_pilot",6,0)

                requestdensityration = DataRefByteGen("sim/weather/sigma",7,0)

                requestliftforce = DataRefByteGen("sim/flightmodel/forces/fnrml_aero",8,0) #In Newtons!

                recsock.sendto(requestagl,(UDP_IP,UDP_PORT))
                recsock.sendto(requestprate,(UDP_IP,UDP_PORT))
                recsock.sendto(requestroll,(UDP_IP,UDP_PORT))
                recsock.sendto(requestrrate,(UDP_IP,UDP_PORT))
                recsock.sendto(requestjoyp,(UDP_IP,UDP_PORT))
                recsock.sendto(requestspeed,(UDP_IP,UDP_PORT))
                recsock.sendto(requestdensityration,(UDP_IP,UDP_PORT))
                recsock.sendto(requestdensityration,(UDP_IP,UDP_PORT))
                RUN = False
                print("stopped")
        elif k.is_pressed('r'):
            if RUN == False:
                #se parado, roda e pede dados novamente ao xplane
                requestagl = DataRefByteGen("sim/cockpit2/gauges/indicators/pitch_AHARS_deg_pilot",1)

                
                requestprate = DataRefByteGen("sim/flightmodel/position/Q",2)

                requestroll = DataRefByteGen("sim/cockpit2/gauges/indicators/roll_AHARS_deg_pilot",3)

                requestrrate = DataRefByteGen("sim/flightmodel/position/P",4)

                requestjoyp = DataRefByteGen("sim/cockpit2/controls/yoke_pitch_ratio",5)

                requestspeed = DataRefByteGen("sim/cockpit2/gauges/indicators/airspeed_kts_pilot",6)

                requestdensityration = DataRefByteGen("sim/weather/sigma",7)

                requestliftforce = DataRefByteGen("sim/flightmodel/forces/fnrml_aero",8) #In Newtons!

                recsock.sendto(requestagl,(UDP_IP,UDP_PORT))
                recsock.sendto(requestprate,(UDP_IP,UDP_PORT))
                recsock.sendto(requestroll,(UDP_IP,UDP_PORT))
                recsock.sendto(requestrrate,(UDP_IP,UDP_PORT))
                recsock.sendto(requestjoyp,(UDP_IP,UDP_PORT))
                recsock.sendto(requestspeed,(UDP_IP,UDP_PORT))
                recsock.sendto(requestdensityration,(UDP_IP,UDP_PORT))
                recsock.sendto(requestdensityration,(UDP_IP,UDP_PORT))

                RUN = True
                print("running")
        
    except:
        pass
    #atualiza referencia, valores antigos e GUI
    C_REF = w1.get()
    controller = w2.get()
    oldcmd = commandp
    oldcmdr = commandr
    pid_pitch = PID( Kp = w3.get(), Ki = w4.get(), Kd = w5.get(), setpoint = C_REF, sample_time = None, output_limits = (-1, 1))
    master.update_idletasks()
    master.update()



    #Plot da superficie

err = np.arange(0,1.1,0.1)
derr = err
z = list()
zR = list()
for elerr in err:
    for elderr in derr:
        #result = outCalc(OutVar,centroids,np.array([[elerr,elderr]]))
        resultR = outCalc(OutVarR,centroidsR,np.array([[elerr,elderr]]))
        result = model.predict(np.array([[elerr,elderr]]))
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