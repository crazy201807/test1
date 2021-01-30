# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 22:59:24 2021

@author: CHEN
"""
def getA(x):
    a1=(x['close'] /x['pre_close'] -1)*10
    a2=(x['open'] /x['pre_close'] -1)*10
    a3=(x['high'] /x['pre_close'] -1)*10
    a4=(x['low'] /x['pre_close'] -1)*10
    a5=(x['amount'] /x['vol'] *10/x['pre_close'] -1)*10
    
#    a = np.concatenate((a1,a2,a3,a4,a5),axis = 0)
    a = a1 
    a = np.array([a]).reshape(-1,1)
    return a
    
    
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
time_start=time.time()

#Data=pd.read_pickle('data_20201218.pkl')
M = len(Data)
N,Fea = Data[0].shape

ObsWin = np.arange(0,20,1) 
M1 = ObsWin.size
trainX = np.zeros((M1*(N-110),41,7))
trainY = np.zeros((M1*(N-110),1))
Data_loc = np.zeros((M1*(N-110),2))

isamp = 0
A = 0
B = 0
succ_num,fail_num =0,0
FeatureNum1 = 20
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
                        A += a@a.transpose()
                        
                        trainY[isamp] = 1
                        succ_num +=1
                        break
                    elif x['close'][j]<= x_down:
                        a=getA(x[ i+FeatureNum1-1:i-1:-1 ])
                        B += a@a.transpose()
                        
                        trainY[isamp] = 0
                        fail_num +=1
                        break
                    else:
                        pass
                isamp += 1
                
A /= succ_num
B /= fail_num
            
  

UA,DA,UA=np.linalg.svd(A)  #A = UA' * DA * UA
UB,DB,UB=np.linalg.svd(B)      
UA1=UA[:,0:3]
UB1=UB[:,0:3]
PA = UA1 @ UA1.transpose()
PB = UB1 @ UB1.transpose()

#x=np.random.rand(1,41)
M1=trainX.shape[0]
predY = np.zeros((M1,2))
predY1=np.zeros((M1,1))
TH=2
for isamp in range(M1):

    index = int(Data_loc[isamp][0])
    i = int(Data_loc[isamp][1])
    if i==0:
        break
#    x0 = np.array(Data[index]['close'])
#    x = x0[i+40:i-1:-1].reshape(1,41) /x0[i+40]
    x=Data[index]
    a=getA(x[ i+FeatureNum1-1:i-1:-1 ])
    a/=np.linalg.norm(a)
    xa = np.linalg.norm(PA @ a)
    xb = np.linalg.norm(PB @ a)
    predY[isamp,0]=xa
    predY[isamp,1]=xb
    if xa/xb>TH:
        predY1[isamp]=1
 
plt.figure(figsize=(20,5))
plt.plot(predY[100:300,0],'r-' ,label='Class A')
plt.plot(predY[100:300,1] ,'g-',label='Class B' )

plt.figure(figsize=(20,5))
plt.plot(predY[100:300,0]/predY[100:300,1],'m-' )
plt.plot(predY1[100:300]==1, 'md')
plt.plot(trainY[100:300]==1, 'b-*')    
#print('xa={0},xb={1}'.format(xa,xb))

sa=np.sum(trainY==1)
sb=np.sum(trainY==0)
Succ_base = sa/(sa+sb) 
Succ_num,Fail_num=0,0     
Re_data = np.zeros((500,120))  
icond=0      
plt.figure(figsize=(18,5))
for isamp in range(trainX.shape[0]): 
    if trainY[isamp]==1:
        Succ_num+=1
        index = int(Data_loc[isamp][0])
        i = int(Data_loc[isamp][1])
        x= np.array(Data[index]['close'])
#        Re_data[icond,:]=x[i+70:i-50:-1]
        icond+=1
#        plt.plot(np.arange(0,70),x[i+70:i:-1]/x[i],'b-')
#        plt.plot(np.arange(70,120),x[i:i-50:-1]/x[i],'g.-')
Succ_cond = Succ_num/(1e-6+Succ_num+Fail_num)
print('Succ_base={0}({1}/{2})'.format(Succ_base,sa,sb))   

a_index=np.where(predY1==1)[0]
succ = np.sum(trainY[a_index]==1)
fail = np.sum(trainY[a_index]==0)
print('Succ_Algo.={0}({1}/{2})'.format(succ/(succ+fail),succ,fail))      

