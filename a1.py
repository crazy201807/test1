import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def evalx(x,a,b):
    Succ,Fail,invalid =0,0,0
    Succ_base,Fail_base,invalid_base =0,0,0
    x1 = np.zeros(N,)
    x2 = np.zeros(N,)
    x3 = np.zeros(N,)
    Flag = np.zeros(N,)
    for i in range(a,b):
        x1[i] = np.mean(x[i-5:i])
        x2[i] = np.mean(x[i-20:i])
        x3[i] = np.mean(x[i-60:i])
        
        x_up = x[i] * 1.10
        x_down = x[i] * 0.90
        for j in range(i+1,i+40):
            if x[j] >= x_up:
                Flag[i] = 1
                break
            elif x[j]<= x_down:
                Flag[i] = 2
                break
            else:
                pass
        
        cond1 = (x[i] > x1[i]*1.0) and (x1[i] > x2[i]*1.0) and (x2[i] > x3[i]*1.0) and (x[i] < x3[i]*1.5)
        cond2 = (x1[i] > x2[i]*1.0) and (x2[i] > x3[i]*1)
        cond3 = (x2[i] > x3[i]*0.99) and (x2[i] < x3[i]*1.01) and (x[i] > x3[i]*1.0) and (x[i] > x1[i]*1.0)\
        and (x[i] > x2[i]*1.0) and (x[i]>x[i-1]) 
        if  1:
            if Flag[i]==1:
                Succ_base +=1
            elif Flag[i]==2:
                Fail_base +=1
            else:
                invalid_base +=1
                
        if  cond3:
            if Flag[i]==1:
                Succ +=1
            elif Flag[i]==2:
                Fail +=1
            else:
                invalid +=1
    result = [[Succ_base, Fail_base,invalid_base],
              [Succ,      Fail,     invalid]]
    i=-1
    x1[i] = np.mean(x[i-5:i])
    x2[i] = np.mean(x[i-20:i])
    x3[i] = np.mean(x[i-60:i])
    cond3 = (x2[i] > x3[i]*0.99) and (x2[i] < x3[i]*1.01) and (x[i] > x3[i]*1.0) and (x[i] > x1[i]*1.0)\
        and (x[i] > x2[i]*1.0) and (x[i]>x[i-1])
       
    return result,cond3
                
(A1,DateSet,Data_baseline, CodeSet,Data) = pd.read_pickle('data_20190823.pkl')
(M,N)=Data.shape

Succ_rate1 = np.zeros((M,))
Succ_rate_base1 = np.zeros((M,))
Flag1 = np.zeros((M,N))

Succ_rate2 = np.zeros((M,))
Succ_rate_base2 = np.zeros((M,))
Flag2 = np.zeros((M,N))

result1=np.zeros((M,2,3))
result2=np.zeros((M,2,3))
cond = np.zeros((M,))
for index in range(M):
    
    x = Data[index]
   
    result1[index],cond[index] = evalx(x,60,N-40)
#    result2[index] = evalx(x,180,240)
    
#print('Succ={0},Fail={1},invalid={2}'.format(Succ,Fail,invalid))
#print('Succ_rate={0}'.format(Succ/(Succ+Fail)))
 
S_rate_base=result1[:,0,0]/(result1[:,0,0]+result1[:,0,1])
S_rate_cond=result1[:,1,0]/(result1[:,1,0]+result1[:,1,1])

    
ObsWin = np.arange(100,300,1) 
plt.figure(figsize=(15,5)) 
plt.plot(S_rate_base,'b.-')
plt.plot(S_rate_cond,'r.-')


#plt.figure(figsize=(15,5))    
#plt.plot(result1[ObsWin,0,0],'r*')
#plt.plot(result1[ObsWin,0,1],'g.')
#plt.figure(figsize=(15,5)) 
#plt.plot(result1[ObsWin,1,0],'r*')
#plt.plot(result1[ObsWin,1,1],'g.')
A=np.sum(result1,axis=0)
print('Base: Succ_rate={0}'.format(A[0,0]/(A[0,0]+A[0,1])))
print('Cond1: Succ_rate={0}'.format(A[1,0]/(A[1,0]+A[1,1])))
print(A)

a1=np.where(S_rate_cond>0.9)
a2=np.where(cond==1)
B_list = np.intersect1d(a1,a2)

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
                
                
                
