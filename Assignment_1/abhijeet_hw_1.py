#Question_1, NAND GATE, Abhijeet Gupta. Student ID : 101680002

import numpy as np      
import random
import matplotlib
import matplotlib.pyplot as plt

w1 = random.uniform(-2.0,2)                                                     # W1 W2 W3 Weights as per the question. 

w2 = random.uniform(-2.0,2)

w3 = random.uniform(-2.0,2)
print(w1,w2,w3)

x1 =[0,0,1,1]
x2 =[0,1,0,1]
t= [1,1,1,0]
print(x1,x2,t)

def sigmoid (x):
  return 1/(1+np.exp(-x))                                                       # To convert Step fuction graph to continous. 
def d_sigmoid(x):    
  return x *(1-x)                                                               # Differentiating Sigmoid 
learning_rate = 0.01

while True: 
 total_error = 0                                                                # Loop for error condition 
 for i in range(0,4):
   d= x1[i]*w1 + x2[i]*w2 +w3       
   y = sigmoid(d)
   print(y)
   error = np.square(t[i]- y)        
   d_error = 2*(t[i]-y)              

   dw1 = learning_rate * d_error * y * (1-y) * x1[i]                            # Formula for Delta w1,w2,w3 (Training Algorithm)
   dw2 = learning_rate * d_error * y * (1-y) * x2[i]
   dw3 = learning_rate * d_error * y * (1-y) * 1 

   w1 = w1 +dw1                                                                 # Adding previous error + Current error 
   w2 = w2 +dw2
   w3 = w3 +dw3

   total_error = total_error + error                                
 print("Total_error: " , total_error)
 print("-----------------------")                                               
 if(total_error <0.001):                                                       # Stopping loop (Condition)

  break
  print("Final Weights are: w1=%f\n w2=%f\n w3=%f\n" %(w1,w2,w3))

fig = plt.figure()                                                              #Plotting Graph With final Weights. 
plt.plot([0,0,1,1],[0,1,0,1],'co')                                              
plt.plot([1],[1],'ms')

x= np.linspace(-3,3,100)                                                            
y= -1*(x*w1+w3)/w2
plt.plot(x,y,color = 'k',linestyle='dashed',linewidth=2, markersize=12)
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()



#Question 2 Abhijeet Gupta Student ID: 101680002

import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt

w1= random.uniform(-2.0,2)

w2= random.uniform(-2.0,2)

w3= random.uniform(-2.0,2)
print(w1,w2,w3)

x1 = list(np.random.uniform(0,3,50))
x2 = list(np.random.uniform(0,3,50))
labels = [0 for i in range(50)]                                                 #target is set to 0

                                                                                #concatenating the list with tother condition of <6<=9
x1 = x1+ list(np.random.uniform(6,9,50))                                        #Plotting the points between 6 to 9 and putting 50 random points 
x2 += list(np.random.uniform(6,9,50))
labels += [1 for i in range(50)]

import pandas as pd                                                             #Importing the panda Library 
df = pd.DataFrame.from_dict({
    "x1": x1,
    "x2": x2,
    "labels": labels
})

def sigmoid (x):                                                                # To convert Step fuction graph to continous.
  return 1/(1+np.exp(-x))

def d_sigmoid(x):                                                               # Differentiating Sigmoid 
  return x *(1-x)

learning_rate = 0.01

while True: 
 total_error = 0
 for i in range(0,100):
   d= df["x1"].tolist()[i]*w1 + df["x2"].tolist()[i]*w2 +w3                     # The Equation
   y = sigmoid(d)
   error = np.square(df["labels"].tolist()[i] - y)                    

   d_error = 2*(df["labels"].tolist()[i]-y)

   dw1 = learning_rate * d_error * y * (1-y) * x1[i]                            #Training algorithm
   dw2 = learning_rate * d_error * y * (1-y) * x2[i]
   dw3 = learning_rate * d_error * y * (1-y) * 1
 
   w1 = w1 +dw1                                                                 # Adding Delta W to get closer to the correct weight 
   w2 = w2 +dw2
   w3 = w3 +dw3

   total_error = total_error + error                                            #Adding the Error to the total error 
 print("Total_error: " , total_error,"y: ", y, end = '/r')
 

 if(total_error <0.01):                                                         #Loop break condition 

    print("Final Weights are:\n w1=%f\n w2=%f\n w3=%f\n" %(w1,w2,w3))
    break

fig = plt.figure()
ax = fig.add_subplot(111)
for x,y,lab in zip(x1,x2,labels):
        ax.scatter(x,y,label=lab)
plt.axis([0,10,0,10])
x= np.linspace(0,10,100)
y= -1*(x*w1+w3)/w2
plt.plot(x,y, )
plt.show()

