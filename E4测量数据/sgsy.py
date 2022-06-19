# %%
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

#读取数据
data = np.loadtxt('data.txt', usecols=(0,1))
rx=data[:,0]
r1=data[:,1]

lambda_0=632.8 #在这里输入基准激光波长

#拟合
def func(x,k):
    return k*x
popt,pcov=curve_fit(func,rx,r1)
lambda_xbest=popt[0]*lambda_0

#计算标准差
s=0
lambda_xi=np.zeros(len(r1))
for i in range(len(rx)):
    s=s+((r1[i]*lambda_0/rx[i])-(lambda_xbest))**2
    lambda_xi[i]=r1[i]*lambda_0/rx[i]
s=np.sqrt(s/(len(r1)-1))

print('第一次绿光波长的拟合值是'+str(lambda_xbest)+'nm')
print('标准差是'+str(s)+'nm')
print('检索并去除误差超过3sigma的数据点....')

##除去3sigma点
ind=np.empty(shape=(0,3))
for i in range(len(r1)):
    err=np.abs(lambda_xi[i]-lambda_xbest)
    if err>=3*s:
        print('除去第'+str(i)+'个点')
        ind=np.append(ind,i)
ind=ind.astype(int)
rx=np.delete(rx,ind)
r1=np.delete(r1,ind)

#第二次拟合
popt,pcov=curve_fit(func,rx,r1)
lambda_xbest=popt[0]*lambda_0

#第二次计算标准差
s=0
lambda_xi=np.zeros(len(r1))
for i in range(len(r1)):
    s=s+((r1[i]*lambda_0/rx[i])-(lambda_xbest))**2
    lambda_xi[i]=r1[i]*lambda_0/rx[i]
s=np.sqrt(s/(len(r1)-1))
print('第二次绿光波长的最佳拟合值是'+str(lambda_xbest)+'nm')
print('标准差是'+str(s)+'nm')


