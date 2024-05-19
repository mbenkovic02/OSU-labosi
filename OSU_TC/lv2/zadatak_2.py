import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('lv2\data.csv', delimiter=",", dtype="str")
data = data[1::]
data = np.array(data, np.float64)
print(f"Amount of people measurred: {len(data)}")
height, weight = data[:, 1], data[:, 2]
plt.scatter(height, weight)
plt.show()
mean = height.mean()
max = height.max()
min = height.min()
height, weight = data[0::50, 1], data[0::50, 2]
plt.scatter(height, weight)
plt.show()
print(f"Max: {max}, min: {min}, mean: {mean}")
men = data[data[:,0] == 1]
women = data[data[:,0] == 0]
print(f"MEN:\n Max: {men[:, 1].max()}, min: {men[:, 1].min()}, mean: {men[:, 1].mean()}")
print(f"WOMEN:\n Max: {women[:, 1].max()}, min: {women[:, 1].min()}, mean: {women[:, 1].mean()}")