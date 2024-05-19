import pandas as pd 
import matplotlib.pyplot as plt

data = pd.read_csv('lv3\data_C02_emission.csv')

# Zadatak pod a)
plt.figure()
data['CO2 Emissions (g/km)'].plot(kind="hist", bins=20)
plt.show()

# Zadatak pod b)
data['Fuel Type'] = data['Fuel Type'].astype('category')
colors = {'Z': 'brown', 'X': 'red', 'E': 'blue', 'D': 'black'}

data.plot.scatter(x="Fuel Consumption City (L/100km)", y="CO2 Emissions (g/km)", c=data["Fuel Type"].map(colors), s=50)
plt.show()

# Zadatak pod c) 
data.boxplot(column='CO2 Emissions (g/km)', by='Fuel Type')
plt.show()

# Zadatak pod d)
fuel_grouped_num = data.groupby('Fuel Type').size()
fuel_grouped_num.plot(kind ='bar', xlabel='Fuel Type', ylabel='Number of vehicles', title='Amount of vehicles by fuel type')
plt.show()

# Zadatak pod e)
cylinder_grouped = data.groupby('Cylinders')['CO2 Emissions (g/km)'].mean()
cylinder_grouped.plot(kind='bar', x=cylinder_grouped.index, y=cylinder_grouped.values, xlabel='Cylinders', ylabel='CO2 emissions (g/km)', title='CO2 emissions by number of cylinders')
plt.show()

