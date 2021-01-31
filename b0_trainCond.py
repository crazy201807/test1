# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 23:15:49 2021

@author: CHEN
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 13:23:46 2021

@author: CHEN
"""

def getA(x):
    return 0

def getCond(x):
    N=len(x)
    x_today = x.iloc[N-1]
    # base prob: 31% for (15% -10%)
    c1 = x.iloc[N-1]['close']>np.mean(x['close'][N-60:N])  # inc 2% succ for (60,20,5)
    c2 = x.iloc[N-1]['close']>x.iloc[N-2]['close'] # 35.2%
    c3 = x.iloc[N-1]['open']<x.iloc[N-2]['close'] and x.iloc[N-1]['close']>x.iloc[N-2]['open'] and x.iloc[N-2]['pct_chg']<0 #36.5%
    
    result1 = c1 and c2 and c3
    
    return result1
    
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
time_start=time.time()

#Data=pd.read_pickle('data_20201218.pkl')
M = len(Data)
N,Fea = Data[0].shape

ObsWin = np.arange(100,300,1) 
M1 = ObsWin.size
trainX = np.zeros((M1*(N-110),41,7))
trainY = np.zeros((M1*(N-110),1))
Data_loc = np.zeros((M1*(N-110),2))

isamp = 0
A = 0
B = 0
succ_num,fail_num =0,0
succ_num1 = 0
FeatureNum1 = 70
A_detail=np.ones((70*4,1))
for index in ObsWin:
    
    if index%100==0:
        print('index={0}'.format(index))
    
    x = Data[index]
    if x is None:
        print('index-{0} is Empty!'.format(index))
        
    else:
    
        N,fea = x.shape
        if N>70:
#            result1[index] = evalx(x,40,N-70)
            for i in range(N-70,40,-20):
                Data_loc[isamp,:]=np.array([index,i])

                x_up = x['close'][i] * 1.15
                x_down = x['close'][i] * 0.9
                for j in range(i-1,i-40-1,-1):
                    if x['high'][j] >= x_up:
                        trainY[isamp] = 1
                        break
                    elif x['close'][j]<= x_down:
                        trainY[isamp] = 2
                        break
                    else:
                        pass
                isamp += 1
                
    
NUM_samp = isamp
trainX = trainX[0:NUM_samp]
trainY = trainY[0:NUM_samp]

succ_base = 0
fail_base = 0 
invalid_base=0
succ_algo = 0
fail_algo = 0
invalid_algo = 0 
for isamp in range(trainX.shape[0]): 
    if 1:
        if trainY[isamp]==1:
            succ_base +=1
        elif trainY[isamp]==2:
            fail_base +=1
        else:
            invalid_base +=1
    index = int(Data_loc[isamp][0])
    i = int(Data_loc[isamp][1])        
    x = Data[index][i+FeatureNum1-1:i-1:-1 ]        
    cond = getCond(x)
    if cond:
        if trainY[isamp]==1:
            succ_algo +=1
        elif trainY[isamp]==2:
            fail_algo +=1
        else:
            invalid_algo+=1
            
        
Succ_base = succ_base/(1e-6+succ_base+fail_base+invalid_base)
Succ_algo = succ_algo/(1e-6+succ_algo+fail_algo+invalid_algo)
print('Succ_base Prob={0:.4f}(succ:{1}/fail:{2}/invalid:{3})'.format(Succ_base,succ_base,fail_base,invalid_base))   
print('Succ_algo Prob={0:.4f}(succ:{1}/fail:{2}/invalid:{3})'.format(Succ_algo,succ_algo,fail_algo,invalid_algo))   
time_end=time.time()

print('totally cost time:',time_end-time_start)
