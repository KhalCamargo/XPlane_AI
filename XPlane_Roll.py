import numpy as np
from sys import getsizeof
from datetime import timedelta
from datetime import datetime
from struct import *
import socket
import pandas as pd
import xlrd as xlrd
import skfuzzy as fuzz
from matplotlib import pyplot as plt
import skfuzzy.control as ctrl
import keyboard as k

firstRun = False

def getData():
    
    if firstRun:
# Gathering all data
        data = pd.read_excel('Dados USAR SO PRA PITCH 3.xlsx')
        data = data.drop(data.index[0:159]) #Removing n-first rows

        data2 = pd.read_excel('Dados USAR SO PRA PITCH 2.xlsx')
        data2 = data.drop(data.index[0:284]) #Removing n-first rows

        data3 = pd.read_excel('Dados USAR SO PRA PITCH.xlsx')
        data3 = data.drop(data.index[0:1003]) #Removing n-first rows

        # Appending data
        data = data.append(data2)
        data = data.append(data3)

        data = data.loc[(abs(data['   _roll,__deg '])) <= 15] #Selecting only pitch within range
        data = data.loc[:,'ErroRoll':]

        ##### New Code
        datas = data.loc[:,['ErroRoll','dErroRoll','CmdRoll']]
        datas.to_pickle('datasetX_Roll.pkl')
    else:
        datas = pd.read_pickle('datasetX_Roll.pkl')

    #datas.loc[:,'ErroRoll':'dErroRoll'] = (datas.loc[:,'ErroRoll':'dErroRoll'] - datas.loc[:,'ErroRoll':'dErroRoll'].min())/(datas.loc[:,'ErroRoll':'dErroRoll'].max()-datas.loc[:,'ErroRoll':'dErroRoll'].min())
    return datas

def normalizeDatas(datas):
    l_datas = datas.copy()
    l_datas.loc[:,'ErroRoll':'dErroRoll'] = (l_datas.loc[:,'ErroRoll':'dErroRoll'] - l_datas.loc[:,'ErroRoll':'dErroRoll'].min())/(l_datas.loc[:,'ErroRoll':'dErroRoll'].max()-l_datas.loc[:,'ErroRoll':'dErroRoll'].min())
    return l_datas
def normalizeNewData(data,datas):    
    datas = datas.loc[:,'ErroRoll':'dErroRoll']
    toClassDF = {'ErroRoll':[data[0][0]],
                 'dErroRoll':[data[0][1]]}
    toClass = pd.DataFrame(toClassDF,columns=['ErroRoll','dErroRoll'])
    toClass = (toClass -  datas.min())/(datas.max()-datas.min())
    toClass['ErroRoll'].loc[(toClass['ErroRoll'] > 1)] = 1
    toClass['ErroRoll'].loc[(toClass['ErroRoll'] < 0)] = 0
    toClass['dErroRoll'].loc[(toClass['dErroRoll'] > 1)] = 1
    toClass['dErroRoll'].loc[(toClass['dErroRoll'] < 0)] = 0
    return toClass.to_numpy()
def generateFuzzySys(datas):
    datasnp = datas.to_numpy()
    n_clusters = 7
    cntrError, uError, u0Error, dError, jmError, pError, fpcError = fuzz.cluster.cmeans(datasnp.T, c=n_clusters, m=2, error=0.005, maxiter=1000, init=None)

    cntrStick = []
    for i in range (0,n_clusters):
        cntrStick.append(cntrError[i,2])
    cntrStick.sort()

    RollStick = ctrl.Consequent(np.arange(-1, 1.001, 0.001), 'Roll Stick')
    """ Saída Stick
    LL - Left Large
    LM - Left Medium
    LS - Left Small
    K - keep
    RS - Right Small
    RM - Right Medium
    RL - Right Large
    """
    RollStick['LL'] = fuzz.trapmf(RollStick.universe,[-2,-1,cntrStick[0],cntrStick[1]])
    RollStick['LM'] = fuzz.trimf(RollStick.universe,[cntrStick[0],cntrStick[1],cntrStick[2]])
    RollStick['LS'] = fuzz.trimf(RollStick.universe,[cntrStick[1],cntrStick[2],cntrStick[3]])
    RollStick['K'] = fuzz.trimf(RollStick.universe,[cntrStick[2],cntrStick[3],cntrStick[4]])
    RollStick['RS'] = fuzz.trimf(RollStick.universe,[cntrStick[3],cntrStick[4],cntrStick[5]])
    RollStick['RM'] = fuzz.trimf(RollStick.universe,[cntrStick[4],cntrStick[5],cntrStick[6]])
    RollStick['RL'] = fuzz.trapmf(RollStick.universe,[cntrStick[5],cntrStick[6],1,2])

 

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

    ## Regras dos extremos
    #R1 = ctrl.Rule(Error['NL'],RollStick['SM'])
    #allRules.append(R1)
    #R2 = ctrl.Rule(Error['PL'],RollStick['DM'])
    #allRules.append(R2)
    #R3 = ctrl.Rule(((Error['NM'] | Error['NS']) & (dError['NL'] | dError['NM'])) | (Error['Z'] & dError['NL']) | (Error['NM'] & dError['NS']),RollStick['SM'])
    #allRules.append(R3)
    #R4 = ctrl.Rule(((Error['PM'] | Error['PS']) & (dError['PL'] | dError['PM'])) | (Error['Z'] & dError['PL']) | (Error['PM'] & dError['PS']),RollStick['DM'])
    #allRules.append(R4)
    #R5 = ctrl.Rule((Error['PS'] & dError['NL']) | (Error['Z'] & dError['NM']) | (Error['NS'] & dError['NS']) | (Error['NM'] & dError['Z']),RollStick['SS'])
    #allRules.append(R5)
    #R6 = ctrl.Rule((Error['PM'] & dError['NL']) | (Error['PS'] & dError['NM']) | (Error['Z'] & dError['NS']) | (Error['NS'] & dError['Z']) | (Error['NM'] & dError['PS']),RollStick['SS'])
    #allRules.append(R6)
    #R7 = ctrl.Rule((Error['PM'] & dError['NM']) | (Error['PS'] & dError['NS']) | (Error['Z'] & dError['Z']) | (Error['NS'] & dError['PS']) | (Error['NM'] & dError['PM']),RollStick['M'])
    #allRules.append(R7)
    #R8 = ctrl.Rule((Error['PM'] & dError['NS']) | (Error['PS'] & dError['Z']) | (Error['Z'] & dError['PS']) | (Error['NS'] & dError['PM']) | (Error['NM'] & dError['PL']),RollStick['DS'])
    #allRules.append(R8)
    #R9 = ctrl.Rule((Error['PM'] & dError['Z']) | (Error['PS'] & dError['PS']) | (Error['Z'] & dError['PM']) | (Error['NS'] & dError['PL']),RollStick['DS'])
    #allRules.append(R9)

    # TEST 1
    #R1 = ctrl.Rule(
    #    (Error['NL'] & dError['NS']) | 
    #    (Error['NS'] & (dError['NL'] | dError['NM'] | dError['NS'] | dError['Z'] ) )  
    #    , RollStick['SS'])
    #allRules.append(R1)
    #R2 = ctrl.Rule(
    #    (Error['NL'] & dError['NM']) | 
    #    (Error['NM'] & (dError['NL'] | dError['NM'] | dError['NS'] | dError['Z'] ) ) | 
    #    (Error['NL'] & dError['Z'])  
    #    , RollStick['SM'])
    #allRules.append(R2)
    #R3 = ctrl.Rule(
    #    (Error['NL'] & dError['NL']), RollStick['SL']
    #    )
    #allRules.append(R3)
    #R4 = ctrl.Rule(
    #    (Error['Z']) | 
    #    (Error['PL'] & dError['NL']) | (Error['PM'] & dError['NM']) | (Error['PS'] & dError['NS']) | (Error['NS'] & dError['PS']) |
    #    (Error['NM'] & dError['PM']) | (Error['NL'] & dError['PL']) |
    #    (Error['NL'] & dError['PS']) | (Error['NL'] & dError['PM']) | (Error['NM'] & dError['PS']) |
    #    (Error['PL'] & dError['NS']) | (Error['PL'] & dError['NM']) | (Error['PM'] & dError['NS']) 
    #     , RollStick['M'])
    #allRules.append(R4)
    #R5 = ctrl.Rule(
    #    (Error['PL'] & dError['PL']) | (Error['PM'] & dError['PL'] ) | (Error['PL'] & dError['PM'] ) | (Error['PL'] & dError['PS'] ) | (Error['PM'] & dError['PM'] ), RollStick['DL']
    #    )
    #allRules.append(R5)
    #R6 = ctrl.Rule(
         
    #    (Error['PM'] & ( dError['PS'] | dError['Z'] ) ) | 
    #    (Error['PL'] & dError['Z']) |
    #    (Error['NS'] & dError['PL']) 
    #    , RollStick['DM'])
    #allRules.append(R6)
    #R7 = ctrl.Rule(
         
    #    (Error['PS'] & (dError['PL'] | dError['PM'] | dError['PS'] | dError['Z'] ) ) | 
    #    (Error['NM'] & dError['PL']) |
    #    (Error['NS'] & dError['PM']) |
    #    (Error['PS'] & dError['NL']) | 
    #    (Error['PM'] & dError['NL']) |
    #    (Error['PS'] & dError['NM'])
    #    , RollStick['DS'])
    #allRules.append(R7)
    #END TEST 1

    #TEST 2
    R1 = ctrl.Rule(
        (Error['NL'] & (dError['NL'] | dError['NM'] | dError['NS']) )
        , RollStick['LL'])

    R2 = ctrl.Rule(
        (Error['NM'] & (dError['NL'] | dError['NM'] | dError['NS'] | dError['Z'] | dError['PS']) ) |
        (Error['NL'] & dError['Z'])
        , RollStick['LM'])

    R3 = ctrl.Rule(
        (Error['NL'] & (dError['PL'] | dError['PM'] | dError['PS']) ) |
        (Error['NM'] & dError['PM']) |
        (Error['NS'] & (dError['NL'] | dError['NM'] | dError['NS'] | dError['Z'] ) )
        , RollStick['LS'])

    R4 = ctrl.Rule(
        (Error['NM'] & dError['PL']) |
        (Error['NS'] & (dError['PS'] | dError['PM'] | dError['PL'] ) ) |
        (Error['Z']) |
        (Error['PS'] & (dError['NL'] | dError['NM']) ) |
        (Error['PM'] & dError['NL'])
        , RollStick['K'])

    R5 = ctrl.Rule(
        (Error['PS'] & (dError['NS'] | dError['Z'] | dError['PS']) ) |
        (Error['PM'] & (dError['NM'] | dError['NS']))
         , RollStick['RS'])

    R6 = ctrl.Rule(
        (Error['PS'] & dError['PM']) |
        (Error['PM'] & (dError['Z'] | dError['PS'] | dError['PM'] )) 
        
        , RollStick['RM'])

    R7 = ctrl.Rule(
        (Error['PL']  ) |
        (dError['PL'] & (Error['PS'] | Error['PM']))
        , RollStick['RL'])
    #END TEST 2

    allRules.append(R1)
    allRules.append(R2)
    allRules.append(R3)
    allRules.append(R4)
    allRules.append(R5)
    allRules.append(R6)
    allRules.append(R7)

    sys = ctrl.ControlSystem(allRules)
    sim = ctrl.ControlSystemSimulation(sys)
    return sim

#class XCHR:
#    def __init__(self,val):
#        self._val = chr(val)
#    def _get(self):
#        return self._val
#    def _set(self,val):
#        self._val = chr(val)
#    val = property(_get,_set)
    
#class XINT:
#    def __init__(self,val):
#        self._val = int(val)
#    def _get(self):
#        return self._val
#    def _set(self,val):
#        self._val = int(val)
#    val = property(_get,_set)
#class XFLT:
#    def __init__(self,val):
#        self._val = np.float(val)
#    def _get(self):
#        return self._val
#    def _set(self,val):
#        self._val = np.float(val)
#    val = property(_get,_set)
        
    
    
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



# Set pitch command
yawDREF = "sim/cockpit2/controls/yoke_pitch_ratio"
yawDREFPadded = yawDREF.ljust(500,"\0")
yawDREFPadded = bytes(yawDREFPadded,'utf-8')
DREFstr = 'DREF\0'
DREFstr = bytes(DREFstr,'utf-8')


def receiveData():
    data = recSock.recv(1024)
    data = data[5:]
    test = len(data)
    id1, altitude, id2, PRate = unpack('ifif',data)
    if id1 == 1:
        out = [altitude,PRate]
    else:
        out = [PRate, altitude]
    return out

def calculate(sim,inputs):
    sim.input['Error'] = inputs[0]
    sim.input['Error Derivative'] = inputs[1]
    sim.compute()
    z = sim.output['Roll Stick']
    return z

def fuzzyInference(Erro,dErro):
    #normalize data
    norm = normalizeNewData([[Erro,dErro]],datas)
    out = calculate(simulation,norm[0])
    return out

def sendCommand(command):
    commandYaw = DREFstr + pack('f',command) + yawDREFPadded
    sendSock.sendto(commandYaw,(UDP_IP,UDP_PORT))


recSock.sendto(requestAGL,(UDP_IP,UDP_PORT))
recSock.sendto(requestPRate,(UDP_IP,UDP_PORT))
RUN = True
datas = getData()

norm_datas = normalizeDatas(datas)

simulation = generateFuzzySys(norm_datas)


C_REF = 5

while(True):
    
    if RUN:
        dataRec = receiveData()
    
        Erro = dataRec[0] - C_REF
        dErro = dataRec[1]

        command = fuzzyInference(Erro,dErro)
        print(f'Error | dError | command ||| {Erro} | {dErro} | {command}')
        sendCommand(command)
    try:  # used try so that if user pressed other than the given key error will not be shown
        if k.is_pressed('s'):  # if key 's' is pressed 
            if RUN:
                requestPRate = RREFstr + pack('ii',0,2) + PRatePadded
                requestAGL = RREFstr + pack('ii',0,1) + AGLDRefPadded
                recSock.sendto(requestAGL,(UDP_IP,UDP_PORT))
                recSock.sendto(requestPRate,(UDP_IP,UDP_PORT))
                RUN = False
                print("STOPPED")
        elif k.is_pressed('r'):
            if RUN == False:
                requestPRate = RREFstr + pack('ii',2,2) + PRatePadded
                requestAGL = RREFstr + pack('ii',2,1) + AGLDRefPadded
                recSock.sendto(requestAGL,(UDP_IP,UDP_PORT))
                recSock.sendto(requestPRate,(UDP_IP,UDP_PORT))
                RUN = True
                print("RUNNING")
    except:
        pass
    
       

    

