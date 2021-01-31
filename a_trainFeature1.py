# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 13:23:46 2021

@author: CHEN
"""

def getA(x):
    N=len(x)
    a1=(x['close'] /x['pre_close'] -1)*10
    a2=(x['open'] /x['pre_close'] -1)*10
    a3=(x['high'] /x['pre_close'] -1)*10
    a4=(x['low'] /x['pre_close'] -1)*10
    a5=(x['amount'] /x['vol'] *10/x['pre_close'] -1)*10
#    a = np.concatenate((a1,a2,a3,a4,a5),axis = 0)
#    a = a1 
    
    C = x.iloc[N-1]['close']  #/x.iloc[N-2]['close']
    O = x.iloc[N-1]['open']
    H = x.iloc[N-1]['high']
    L = x.iloc[N-1]['low']
    A = x.iloc[N-1]['amount']/x.iloc[N-1]['vol']*10
    
    MA60 = np.mean(x['close'][N-60:N])
    MA20 = np.mean(x['close'][N-20:N])
    MA10 = np.mean(x['close'][N-10:N])
    MA5  = np.mean(x['close'][N-5:N])
    
    MA60A = np.mean(x['close'][N-70:N-10])
    MA20A = np.mean(x['close'][N-30:N-10])
    MA10A = np.mean(x['close'][N-20:N-10])
    MA5A  = np.mean(x['close'][N-15:N-10])
    
    a = np.array([C/O,H/O,L/O,MA5/MA5A,MA10/MA10A,MA20/MA20A,MA60/MA60A ])
    
#    a1 = x['close']/np.array(x['close'])[0]
#    a2 = x['open']/np.array(x['close'])[0]
#    a3 = x['high']/np.array(x['close'])[0]
#    a4 = x['low']/np.array(x['close'])[0]
#    a = np.concatenate((a1,a2,a3,a4),axis = 0)
#    a = a1
#    a = np.array([a]).reshape(-1,1)

    return a

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
            for i in range(N-70,40,-10):
                Data_loc[isamp,:]=np.array([index,i])

                x_up = x['close'][i] * 1.15
                x_down = x['close'][i] * 0.9
                for j in range(i-1,i-40-1,-1):
                    if x['high'][j] >= x_up:
                        
                        a=getA(x[ i+FeatureNum1-1:i-1:-1 ])
                        if np.mean(abs(a-B))>0:
                            succ_num1 +=1
                            alpha = 1/succ_num1
                            A = (1-alpha)*A + alpha*a
#                            A_detail=np.concatenate((A_detail,a),axis=1)
                        trainY[isamp] = 1
                        
                        break
                    elif x['close'][j]<= x_down:
                        fail_num +=1
                        alpha = 1/fail_num
                        a=getA(x[ i+FeatureNum1-1:i-1:-1 ])
                        B = (1-alpha)*B + alpha*a
                        
                        trainY[isamp] = 2
                        break
                    else:
                        pass
                isamp += 1
                
#A /= succ_num
#B /= fail_num
plt.figure(figsize=(20,5))
plt.plot(A ,'r*-',label='Good Case({0})'.format(succ_num1));
plt.plot(B, 'g.-',label='bad  Case({0})'.format(fail_num));
plt.grid()
plt.legend()
    
NUM_samp = isamp
trainX = trainX[0:NUM_samp]
trainY = trainY[0:NUM_samp]

M1=trainX.shape[0]
predY1=np.zeros((M1,1))
pred = np.zeros((M1,2))
for isamp in range(M1):
    index = int(Data_loc[isamp][0])
    i = int(Data_loc[isamp][1])
    if i==0:
        break
    x=Data[index]
    a=getA(x[ i+FeatureNum1-1:i-1:-1 ])
    TA = np.mean(np.abs(a-A))
    TB = np.mean(np.abs(a-B))
    pred[isamp,0]=TA
    pred[isamp,1]=TB
    if TA<TB-0.01 :
        predY1[isamp]=1
 
plt.figure(figsize=(20,5))
plt.plot(pred[100:300,0],'r-' ,label='dis from A')
plt.plot(pred[100:300,1] ,'g-',label='dis from B' )
plt.legend()
plt.figure(figsize=(20,5))
plt.plot(predY1[100:300]==1, 'md')
plt.plot(trainY[100:300]==1, 'b-.')    
#print('xa={0},xb={1}'.format(xa,xb))

sa=np.sum(trainY==1)
sb=np.sum(trainY==2)
Succ_base = sa/(sa+sb) 
Succ_num,Fail_num=0,0     
Re_data = np.zeros((500,120))  
icond=0      
plt.figure(figsize=(18,5))

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

a_index=np.where(predY1==1)[0]
succ = np.sum(trainY[a_index]==1)
fail = np.sum(trainY[a_index]==2)
invalid = np.sum(trainY[a_index]==0)
print('Succ_Algo_subSpace.={0:.4f}(succ:{1}/fail:{2}/invalid:{3})'.format(succ/(1e-6+succ+fail+invalid),succ,fail,invalid))      