# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 22:43:25 2020

@author: CHEN
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
time_start=time.time()

Data=pd.read_pickle('data_20201227.pkl')
#plt.plot(np.array(Data[0]['close'])[::-1])
def policy(x):
    pass

def evalx(x,a,b,PLOT_ON=0,MODE=0):
    Short = 5
    Mid = 20
    Long = 60
    N,Fea = x.shape
    Succ,Fail,invalid =0,0,0
    Succ_base,Fail_base,invalid_base =0,0,0
    x1 = np.zeros(N,)
    x2 = np.zeros(N,)
    x3 = np.zeros(N,)
    dif_x = np.zeros(N,)
    dea = np.zeros(N,)
    y1 = np.zeros(N,)
    y2 = np.zeros(N,)
    y3 = np.zeros(N,)
    Flag = np.zeros(N,)
    if MODE==0:
        for i in range(b,a,-1):
            
            x1[i] = np.mean(x['close'][i:i+Short])
            x2[i] = np.mean(x['close'][i:i+Mid])
            x3[i] = np.mean(x['close'][i:i+Long])
            x_min = np.min(x['low'][i:i+40])
            sigma = np.std(x['close'][i:i+Mid])
#            x_max = np.max(x['high'][i:i+40])
            
#            x1[i]=x['close'][i] * 0.95 + x1[i+1] *0.05
#            x2[i]=x['close'][i] * 0.9 + x2[i+1] *0.1
#            x3[i]=x['close'][i] * 0.8 + x3[i+1] *0.2
#            if i>N-60:
#                continue
            
#            dif_x[i] = np.mean(x['close'][i:i+10]) - np.mean(x['close'][i:i+20])
#            dea[i] = np.mean(dif_x[i:i+5])
            
#            y1[i] = np.mean(x['vol'][i:i+Short])
#            y2[i] = np.mean(x['vol'][i:i+Mid])
#            y3[i] = np.mean(x['vol'][i:i+Long])
            
            
            x_up = x['close'][i] * 1.15
            x_down = x['close'][i] * 0.90
            for j in range(i-1,i-40-1,-1):
                if x['high'][j] >= x_up:
                    Flag[i] = 1
                    break
                elif x['close'][j]<= x_down:
                    Flag[i] = 2
                    break
                else:
                    pass
            
           
            cond3 = (x2[i] > x3[i]*1.0) and (x2[i] < x3[i]*1.01)  \
                 and (x['close'][i] > max(x3[i],x2[i],x1[i]) ) \
                 and (x['close'][i]>1.05*x['close'][i+10])  and abs(x['pct_chg'][i])<5\
                 and (x['close'][i]<1.2*x_min) \
                 and (x['vol'][i]<1.01*x['vol'][i+1])
#                 and sigma/x['close'][i] < 0.02
#                 and (y1[i]>y2[i])
#                 and (dif_x[i]>dea[i]) #and (dif_x[i]<dea[i]*1.01)
#            cond3 = (x['close'][i]>1.05*x['close'][i+10])  
            cond3 = np.mean(x['close'][i+0:i+20]) > np.mean(x['close'][i+0:i+60])\
                and np.mean(x['close'][i+0:i+20]) <1.01* np.mean(x['close'][i+0:i+60]) \
                and np.mean(x['close'][i+5:i+25]) > np.mean(x['close'][i+5:i+65])\
                and np.mean(x['close'][i+5:i+25]) <1.01* np.mean(x['close'][i+5:i+65])\
                and (x['close'][i] > max(x3[i],x2[i],x1[i]) )
#                and (x['close'][i]<1.2*x_min)\
#                and (x['vol'][i]<1.01*x['vol'][i+1])
            
            if  1:
                if Flag[i]==1:
                    Succ_base +=1
                elif Flag[i]==2:
                    Fail_base +=1
                else:
                    invalid_base +=1
                    
            if  cond3:
                if PLOT_ON==1:
                    plt.plot(np.array(x['close'])[i:i-40:-1]/x['close'][i])
                if Flag[i]==1:
                    Succ +=1
                elif Flag[i]==2:
                    Fail +=1
                else:
                    invalid +=1
        result = [[Succ_base, Fail_base,invalid_base],
                  [Succ,      Fail,     invalid]]
        return result
    elif MODE==1:
        
        i=0
        x1[i] = np.mean(x['close'][i:i+Short])
        x2[i] = np.mean(x['close'][i:i+Mid])
        x3[i] = np.mean(x['close'][i:i+Long])
        x_min = np.min(x['low'][i:i+40])
        
        cond3 = np.mean(x['close'][i+0:i+20]) > np.mean(x['close'][i+0:i+60])\
                and np.mean(x['close'][i+0:i+20]) <1.01* np.mean(x['close'][i+0:i+60]) \
                and np.mean(x['close'][i+5:i+25]) > np.mean(x['close'][i+5:i+65])\
                and np.mean(x['close'][i+5:i+25]) <1.01* np.mean(x['close'][i+5:i+65])\
                and (x['close'][i] > max(x3[i],x2[i],x1[i]) )\
#                and (x['close'][i]<1.2*x_min)
#                and (x['vol'][i]<1.01*x['vol'][i+1])
                       
        return cond3
                
M = len(Data)
N,Fea = Data[0].shape


Succ_rate1 = np.zeros((M,))
Succ_rate_base1 = np.zeros((M,))
Flag1 = np.zeros((M,N))

Succ_rate2 = np.zeros((M,))
Succ_rate_base2 = np.zeros((M,))
Flag2 = np.zeros((M,N))

result1=np.zeros((M,2,3))
result2=np.zeros((M,2,3))
cond = np.zeros((M,))
ObsWin = np.arange(0,50,1) 
for index in ObsWin:
    if index%100==0:
        print('index={0}'.format(index))
    
    x = Data[index]
    if x is None:
        print('index-{0} is Empty!'.format(index))
        
    else:
    
        N,fea = x.shape
        if N>60:
            result1[index] = evalx(x,40,N-60)
#    result2[index] = evalx(x,300,600)
    
    
#for index in range(M):
#    x = Data[index]
#    if x is None:
#        print(index)
#    else:
#        cond[index] = evalx(x,60,N-40,PLOT_ON=0,MODE=1)
    
#print('Succ={0},Fail={1},invalid={2}'.format(Succ,Fail,invalid))
#print('Succ_rate={0}'.format(Succ/(Succ+Fail)))
 
S_rate_base1=result1[:,0,0]/(1e-6+result1[:,0,0]+result1[:,0,1])
S_rate_cond1=result1[:,1,0]/(1e-6+result1[:,1,0]+result1[:,1,1])

#S_rate_base2=result2[:,0,0]/(1e-6+result2[:,0,0]+result2[:,0,1])
#S_rate_cond2=result2[:,1,0]/(1e-6+result2[:,1,0]+result2[:,1,1])
    

plt.figure(figsize=(12,5)) 
plt.plot(ObsWin,S_rate_base1[ObsWin],'b.-')
plt.plot(ObsWin,S_rate_cond1[ObsWin],'r.-')
#plt.plot(ObsWin,S_rate_base2[ObsWin],'b*--')
#plt.plot(ObsWin,S_rate_cond2[ObsWin],'r*--')


#plt.figure(figsize=(15,5))    
#plt.plot(result1[ObsWin,0,0],'r*')
#plt.plot(result1[ObsWin,0,1],'g.')
#plt.figure(figsize=(15,5)) 
#plt.plot(result1[ObsWin,1,0],'r*')
#plt.plot(result1[ObsWin,1,1],'g.')
A=np.sum(result1,axis=0)
print('Base: Succ_rate={0}'.format(A[0,0]/(1e-6+A[0,0]+A[0,1])))
print('Cond1: Succ_rate={0}'.format(A[1,0]/(1e-6+A[1,0]+A[1,1])))
np.set_printoptions(suppress=True)
print(A)

a1=np.where(S_rate_cond1>0.9)[0]
#a2=np.where(cond==1)[0]
#B_list = np.intersect1d(a1,a2)

if 0:
    plt.figure()
    for index in a1:
        print(index)
        x = Data[index]
        evalx(x,40,N-60,PLOT_ON=1)



if 0:  
    ObsWin = np.arange(100,300,1)        
    plt.figure(figsize=(12,10))
    plt.subplot(211)
    plt.plot(x[ObsWin],'k-')
    plt.plot(x_5[ObsWin],'r-')
    plt.plot(x_20[ObsWin],'g-')
    plt.plot(x_60[ObsWin],'b-')
    plt.grid()
    plt.subplot(212)
    plt.plot(Flag[0,ObsWin])
    plt.grid()
    
    plt.figure()
    for i in range(N):
        if trainY[i]==1:
            plt.plot(x1[i],x2[i],'r.')
        elif trainY[i]==2:
            plt.plot(x1[i],x2[i],'g.')
        else:
            plt.plot(x1[i],x2[i],'k.')
    
    xmin = np.min(x)
    xmax = np.max(x)
    plt.xlim([xmin,xmax])
    plt.ylim([xmin,xmax])
    plt.plot(np.arange(xmin,xmax,0.01),np.arange(xmin,xmax,0.01))
    plt.grid()
                
    
time_end=time.time()
print('totally cost',time_end-time_start)
