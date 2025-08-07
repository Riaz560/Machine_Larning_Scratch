import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read data
Data = pd.read_csv("data.csv")
x_values = Data["YearsExperience"].values
y_values = Data["Salary"].values

# Plot data
'''plt.scatter(x_values, y_values)
plt.xlabel("YearsExperience")
plt.ylabel("Salary")
plt.savefig("scatter_plot.png")'''

def cost_function(m,c,x,y):
  """docstring for cost_function"""
  n=len(x)
  total_err=0.0
  for i in range(n):
    y_cal=(m*x[i]+c)
    total_err=(y[i]-(y_cal)) ** 2
  return total_err/n

def gredant_Desent(x,y,m_currant=0,c_currant=0,epochs=100,larning_rate=0.01):
  """docstring for gredant_Desent"""
  n=len(x)
  m=m_currant
  c=c_currant
  cost_history=[]
  for _ in range(epochs):
    m_gradient=0
    c_gradient=0
    for i in range(n):
      # TODO: write code...
      y_prad=m*x[i]+c
      err=y_prad-y[i]
      m_gradient+=(2/n)*(err*x[i])
      c_gradient+=(2/n)*err
    m=m-larning_rate*m_gradient
    c=c-larning_rate*c_gradient
    carrant_cost=cost_function(x,y,m,c)
    cost_history.append(carrant_cost)
    return m,c,cost_history
inisial_m=0
inisial_c=0
result_m,result_c,cost_history=gredant_Desent(x_values,y_values,inisial_m,inisial_c)
print(result_m)
print(result_c)
print(cost_history)