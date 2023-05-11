# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary library.

2.Load the text file in the compiler.

3.Plot the graphs using sigmoid , costfunction and gradient descent.

4.Predict the values.

5.End the Program.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: P NAVEEN KUMAR
RegisterNumber:  212222230092 
*/
```
```


import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data = np.loadtxt("/content/ex2data1 (2).txt",delimiter=',')
x=data[:,[0,1]]
y=data[:,2]

x[:5]

y[:5]

plt.figure()
plt.scatter(x[y==1][:,0],x[y==1][:,1],label="Admitted")
plt.scatter(x[y==0][:,0],x[y==0][:,1],label="Not admitted")
plt.xlabel("Exam 1 Score")
plt.ylabel("Exam 2 Score")
plt.legend()
plt.show()

def signoid(z):
  return 1/(1+np.exp(-z))
  
plt.plot()
x_plot=np.linspace(-10,10,100)
plt.plot(x_plot,signoid(x_plot))
plt.show()

def costFunction(theta,x,y):
  h=signoid(np.dot(x,theta))
  J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/x.shape[0]
  grad=np.dot(x.T,h-y)/x.shape[0]
  return J,grad
  
x_train=np.hstack((np.ones((x.shape[0],1)),x))
theta=np.array([0,0,0])
J,grad=costFunction(theta,x_train,y)
print(J)
print(grad)

x_train=np.hstack((np.ones((x.shape[0],1)),x))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,x_train,y)
print(J)
print(grad)

def cost(theta,x,y):
  h=signoid(np.dot(x,theta))
  J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/x.shape[0]
  return J

def gradient(theta,x,y):
  h=signoid(np.dot(x,theta))
  grad=np.dot(x.T,h-y)/x.shape[0]
  return grad

x_train=np.hstack((np.ones((x.shape[0],1)),x))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(x_train,y),method="Newton-CG",jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,x,y):
  x_min,x_max=x[:,0].min()-1,x[:,0].max()+1
  y_min,y_max=x[:,1].min()-1,x[:,1].max()+1
  xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
  x_plot=np.c_[xx.ravel(),yy.ravel()]
  x_plot=np.hstack((np.ones((x_plot.shape[0],1)),x_plot))
  y_plot=np.dot(x_plot,theta).reshape(xx.shape)

  plt.figure()
  plt.scatter(x[y==1][:,0],x[y==1][:,1],label="Admitted")
  plt.scatter(x[y==0][:,0],x[y==0][:,1],label="Not Admitted")
  plt.contour(xx,yy,y_plot,levels=[0])
  plt.xlabel("Exam 1 Score")
  plt.ylabel("Exam 2 Score")
  plt.legend()
  plt.show()

plotDecisionBoundary(res.x,x,y)

prob=signoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,x):
  x_train=np.hstack((np.ones((x.shape[0],1)),x))
  prob=signoid(np.dot(x_train,theta))
  return (prob>=0.5).astype(int)
  
np.mean(predict(res.x,x)==y)


```

## Output:
## 1. Array Value of x:
![image](https://github.com/Naveen22009215/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119401470/9497e5ee-71c3-4f37-ba1b-6851c603b49c)

## 2. Array Value of y:
![image](https://github.com/Naveen22009215/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119401470/f8c99a13-38e4-4832-a128-f910fbfce778)

## 3. Exam 1 - score graph:
![image](https://github.com/Naveen22009215/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119401470/9a6e4ce7-68b3-45da-856d-7f569018576b)

## 4. Sigmoid function graph:
![image](https://github.com/Naveen22009215/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119401470/c4901306-4c60-4fcd-bc58-261649021136)

## 5. X_train_grad value:
![image](https://github.com/Naveen22009215/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119401470/a76b4aaa-0d81-4c42-8515-02b2440ae620)

## 6. Y_train_grad value:
![image](https://github.com/Naveen22009215/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119401470/4d02f346-2015-4d7a-b8a3-db25a16c60f5)

## 7. Print res.x:
![image](https://github.com/Naveen22009215/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119401470/c83dc3c8-feb2-4f31-8820-0d6bb46fdd6c)

## 8. Decision boundary - graph for exam score:
![image](https://github.com/Naveen22009215/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119401470/cb30530a-5ead-4e07-8419-91945dd2162b)

## 9. Proability value:
![image](https://github.com/Naveen22009215/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119401470/2b2d2ff3-6c70-471c-b120-58ba4933e93c)

## 10. Prediction value of mean
![image](https://github.com/Naveen22009215/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119401470/74c30ed6-db21-4766-946c-aeddf6f3e8d8)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

