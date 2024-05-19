"""Zadatak 3.4.2 Napišite programski kod koji ce prikazati sljede ´ ce vizualizacije: ´
a) Pomocu histograma prikažite emisiju C02 plinova. Komentirajte dobiveni prikaz. ´
b) Pomocu dijagrama raspršenja prikažite odnos izme ´ du gradske potrošnje goriva i emisije ¯
C02 plinova. Komentirajte dobiveni prikaz. Kako biste bolje razumjeli odnose izmedu¯
velicina, obojite to ˇ ckice na dijagramu raspršenja s obzirom na tip goriva. ˇ
c) Pomocu kutijastog dijagrama prikažite razdiobu izvangradske potrošnje s obzirom na tip ´
goriva. Primjecujete li grubu mjernu pogrešku u podacima? ´
d) Pomocu stup ´ castog dijagrama prikažite broj vozila po tipu goriva. Koristite metodu ˇ
groupby.
e) Pomocu stup ´ castog grafa prikažite na istoj slici prosje ˇ cnu C02 emisiju vozila s obzirom na ˇ
broj cilindara."""


import pandas as pd 
import matplotlib.pyplot as plt

data = pd.read_csv('lv3\data_C02_emission.csv')

#a)

plt.figure()
data ['CO2 Emissions (g/km)'].plot( kind ="hist", bins = 20 )
plt.show()

#b)

data['Fuel Type'] = data['Fuel Type'].astype('category')
colors = {'Z': 'brown', 'X': 'red', 'E': 'blue', 'D': 'black'}

data.plot.scatter(x="Fuel Consumption City (L/100km)", y="CO2 Emissions (g/km)", c=data["Fuel Type"].map(colors), s=50)
plt.show()

#c)

data.boxplot(column='CO2 Emissions (g/km)', by='Fuel Type')
plt.show()


#d)

fuel_grouped_num = data.groupby('Fuel Type').size()
fuel_grouped_num.plot(kind ='bar', xlabel='Fuel Type', ylabel='Number of vehicles', title='Amount of vehicles by fuel type')
plt.show()

#group_by_fuel.plot(kind ='bar', xlabel='Fuel Type', ylabel='Number of vehicles', title='Amount of vehicles by fuel type')
#plt.show()

#e)

cylinder_co2 = data.groupby("Cylinders")['CO2 Emissions (g/km)'].mean()
cylinder_co2.plot(kind ='bar', xlabel='Cylinders', ylabel='CO2 emissions', title='Average emissions based on cylinder count')
plt.show()