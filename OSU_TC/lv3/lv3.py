import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# Primjer 3.1

s1 = pd.Series(['crvenkapica', 'baka', 'majka', 'lovac', 'vuk'])
print(s1)

s2 = pd.Series(5., ['a', 'b', 'c', 'd', 'e'], name = 'ime_objekta')
print(s2)
print(s2['b'])

s3 = pd.Series(np.random.randn(5))
print(s3)
print(s3[3])

# Primjer 3.2

data = {
    'country': ['Italy', 'Spain', 'Greece', 'France', 'Portugal'],
    'population': [59, 47, 11, 68, 10],
    'code': [39, 34, 30, 33, 351]
}

countries = pd.DataFrame(data, columns=['country', 'population', 'code'])

print(countries)

# Primjer 3.4 

data = pd.read_csv('data_C02_emission.csv')

print(len(data))
print(data)

print(data.head(5))
print(data.tail(3))
print(data.info())
print(data.describe())

print(data.max())
print(data.min())

# Primjer 3.5

data = pd.read_csv('data_C02_emission.csv')

print(data['Cylinders'])
print(data.Cylinders)

print(data[['Cylinders', 'Model']])

print(data.iloc[2:6, 2:7])
print(data.iloc[:, 2:5])
print(data.iloc[:, [0,4,7]])

print(data.Cylinders > 6)
print(data[data.Cylinders > 6])
print(data[(data['Cylinders'] == 4) & (data['Engine Size (L)'] > 2.4)].Model)

data['jedinice'] = np.ones(len(data))
data['large'] = (data['Cylinders'] > 10)

print(data)

# Primjer 3.6

data = pd.read_csv('data_C02_emission.csv')

new_data = data.groupby('Cylinders')
print(new_data.count())
print(new_data.size())
print(new_data)
# print(new_data.sum())
# print(new_data.mean())

# Primjer 3.7

data = pd.read_csv('data_C02_emission.csv')

print(data.isnull().sum())
data.dropna(axis=0)
data.dropna(axis=1)
data.drop_duplicates()

data = data.reset_index(drop=True)
print(data)

# Primjer 3.8

data = pd.read_csv('data_C02_emission.csv')

plt.figure()
data['Fuel Consumption City (L/100km)'].plot(kind='hist', bins = 20)
plt.figure()
data['Fuel Consumption City (L/100km)'].plot(kind='box')
plt.show()

# Primjer 3.9

data = pd.read_csv('data_C02_emission.csv')

grouped = data.groupby('Cylinders')
grouped.boxplot(column=['CO2 Emissions (g/km)'])

data.boxplot(column=['CO2 Emissions (g/km)'], by='Cylinders')
plt.show()

# Primjer 3.10

data = pd.read_csv('data_C02_emission.csv')

data.plot.scatter(x='Fuel Consumption City (L/100km)', y='Fuel Consumption Hwy (L/100km)', c='Engine Size (L)', s=50)
plt.show()

# Primjer 3.11

data = pd.read_csv('data_C02_emission.csv')

print(data.corr(numeric_only=True))
