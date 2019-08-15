'''
Created on Aug 12, 2019

@author: chen
'''
import tensorflow as tf
import  numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class AI_models():
    def __init__(self,Data):
        self.Data=Data
        Nin=Data['trainX'].shape[1]
        Nout=Data['trainY'].shape[1]
        
        self.g1=tf.Graph()
        self.sess=tf.Session(graph=self.g1)
        with self.g1.as_default():
            self.Nin=Nin 
            self.Nout=Nout
            self.x=tf.placeholder(tf.float32, [None,Nin], name='x-input')
            self.y_=tf.placeholder(tf.float32, [None,Nout], name='y-input')
#             self.y,self.vars=self.inference()
#             self.train()
        
        
    def inference(self,x):
        print('run AI_model inference')
#         g1=self.g1
#         with g1.as_default():
        W=tf.get_variable(name='W', shape=[self.Nin,self.Nout], 
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer(),
            trainable=True)
        b=tf.get_variable(name='b', shape=[self.Nout], dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer(),
            trainable=True)  
        y=tf.matmul(x,W)+b
        return y,[W,b]
    
    def train(self):
        X1=self.Data['trainX']
        Y1=self.Data['trainY']
        X2=self.Data['testX']
        Y2=self.Data['testY']
        
        N_samples=X1.shape[0]
        batch_size=32
        
        with self.g1.as_default():
            self.y,self.vars=self.inference(self.x)
            self.loss=tf.reduce_mean(tf.square(self.y-self.y_),name='loss_mse')
            loss=self.loss
            optimizer=tf.train.AdamOptimizer(0.01).minimize(loss)
    #         with self.sess as sess:
            sess=self.sess
            sess.run(tf.global_variables_initializer())
            step_vec=[]
            loss1_vec=[]
            loss2_vec=[]
            for istep in range(10001):
                rand_index = np.random.choice(N_samples, size=batch_size)
                
                if istep%1000==0:
                    loss1=sess.run(loss,feed_dict={self.x:X1[rand_index],self.y_:Y1[rand_index]})
                    loss2=sess.run(loss,feed_dict={self.x:X2,self.y_:Y2})
                    print('step={0},train loss={1},test loss={2}'.
                          format(istep,loss1,loss2))
                    step_vec.append(istep+1)
                    loss1_vec.append(loss1)
                    loss2_vec.append(loss2)
                sess.run(optimizer,feed_dict={self.x:X1[rand_index],self.y_:Y1[rand_index]})
        plt.plot(step_vec,loss1_vec,'b.-',label='train')
        plt.plot(step_vec,loss1_vec,'r.-',label='test')
    def get_train_vars(self):
        return [(v.name,v.eval(session=self.sess)) for v in self.vars]
    
    def model_test(self,testX,testY):
        return self.sess.run(self.loss,feed_dict={self.x:testX,self.y_:testY})
    
    def model_pred(self,testX):
        return self.sess.run(self.y,feed_dict={self.x:testX})
    
    def model_save(self):
        tf.summary.merge_all()
        summary_witer=tf.summary.FileWriter('/home/chen/eclipse-workspace/testPython/model/1',self.g1)
        summary_witer.add_graph(self.g1)
        saver=tf.train.Saver(self.vars)
        path=saver.save(self.sess,'/home/chen/eclipse-workspace/testPython/model/m1.ckpt',
                        write_meta_graph=False,write_state=False)
        print('saved {0}'.format(path))
        return path
    
    def pred_offline(self,path,X):
        
        g2=tf.Graph()
        with g2.as_default():
            with tf.Session(graph=g2) as sess2:
                x=tf.placeholder(tf.float32, [None,4], name='x-input')
                y,vars=self.inference(x)
                saver=tf.train.Saver(vars)
                saver.restore(sess2, path)
#                 print(vars)
                ty_pred=sess2.run(y,feed_dict={x:X})
        return ty_pred
        
        
    def __del__(self):
        print('run __del__')
        if self.sess!=None:
            self.sess.close()
        
        
class DNN(AI_models):
    def __init__(self,Data):
        AI_models.__init__(self,Data)
#         print(self)
#         print('DNN')
        
    def inference(self,x):
#         g1=self.g1
        print('run DNN inference')
#         with g1.as_default():
        W=tf.get_variable(name='W', shape=[self.Nin,20], 
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer(),
            trainable=True)
        b=tf.get_variable(name='b', shape=[20], dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer(),
            trainable=True)  
        y=tf.matmul(x,W)+b
        y=tf.sigmoid(y)
        
        W2=tf.get_variable(name='W2', shape=[20,self.Nout], 
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer(),
            trainable=True)
        b2=tf.get_variable(name='b2', shape=[self.Nout], dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer(),
            trainable=True) 
        y=tf.matmul(y,W2)+b2
        
        return y,[W,b,W2,b2]


                
if __name__=='__main__':
    Feature=np.random.randn(6000,4)
    label=np.zeros((6000,1))
    for i in range(6000):
        label[i,0]=Feature[i,0]**2+Feature[i,1]*2+Feature[i,2]*3+Feature[i,3]*4
#     label=np.sum(Feature,axis=1).reshape(6000,1)
#     A=np.concatenate((Feature,label),axis=1)
    Data=dict()
    Data['trainX']=Feature[0:5000]
    Data['trainY']=label[0:5000]
    Data['testX']=Feature[5000:6000]
    Data['testY']=label[5000:6000]
#     AI_models(Data)
    m1=DNN(Data)
#     m1.train()
#     loss1=m1.model_test(Data['testX'], Data['testY'])
#     path=m1.model_save()
#     print(loss1)
    path='/home/chen/eclipse-workspace/testPython/model/m1.ckpt'
    ty_pred=m1.pred_offline(path, Data['testX'])
    testmse=np.mean((ty_pred-Data['testY'])**2)
    print(testmse)

    
    
#     vars=m1.get_train_vars()
#     plt.show()
#     print(vars)
            
        
        
    
        
        
