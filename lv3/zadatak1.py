"""Zadatak 3.4.1 Skripta zadatak_1.py ucitava podatkovni skup iz ˇ data_C02_emission.csv.
Dodajte programski kod u skriptu pomocu kojeg možete odgovoriti na sljede ´ ca pitanja: ´
a) Koliko mjerenja sadrži DataFrame? Kojeg je tipa svaka velicina? Postoje li izostale ili ˇ
duplicirane vrijednosti? Obrišite ih ako postoje. Kategoricke veli ˇ cine konvertirajte u tip ˇ
category.
b) Koja tri automobila ima najvecu odnosno najmanju gradsku potrošnju? Ispišite u terminal: ´
ime proizvoda¯ ca, model vozila i kolika je gradska potrošnja. ˇ
c) Koliko vozila ima velicinu motora izme ˇ du 2.5 i 3.5 L? Kolika je prosje ¯ cna C02 emisija ˇ
plinova za ova vozila?
d) Koliko mjerenja se odnosi na vozila proizvoda¯ ca Audi? Kolika je prosje ˇ cna emisija C02 ˇ
plinova automobila proizvoda¯ ca Audi koji imaju 4 cilindara? ˇ
e) Koliko je vozila s 4,6,8. . . cilindara? Kolika je prosjecna emisija C02 plinova s obzirom na ˇ
broj cilindara?
f) Kolika je prosjecna gradska potrošnja u slu ˇ caju vozila koja koriste dizel, a kolika za vozila ˇ
koja koriste regularni benzin? Koliko iznose medijalne vrijednosti?
g) Koje vozilo s 4 cilindra koje koristi dizelski motor ima najvecu gradsku potrošnju goriva? ´
h) Koliko ima vozila ima rucni tip mjenja ˇ ca (bez obzira na broj brzina)? ˇ
i) Izracunajte korelaciju izme ˇ du numeri ¯ ckih veli ˇ cina. Komentirajte dobiveni rezultat."""


import pandas as pd
import matplotlib . pyplot as plt

data = pd.read_csv("lv3\data_C02_emission.csv")

#a)
print(f"Broj mjerenja:", len(data))




print(data.info())    #ili for col in data.columns:
                      #   print(f"{col} has a type of {data[col].dtype}")



print(f"Null redci:\n", data.isnull().sum())
print(f"Duplicirani redci:\n", data.duplicated().sum())

data.dropna(axis =0)
data.drop_duplicates()


for col in data.columns:
    if data[col].dtype == "object":
        data[col] = data[col].astype('category')


#b)

najveci_gradski_potrosaci = data.nlargest(3, 'Fuel Consumption City (L/100km)')
najmanji_gradski_potrosaci = data.nsmallest(3, 'Fuel Consumption City (L/100km)')

print(f"Najveci gradski potrosaci:\n", najveci_gradski_potrosaci[['Make', 'Model', 'Fuel Consumption City (L/100km)']])
print(f"Najmanji gradski potrosaci:\n", najmanji_gradski_potrosaci[['Make', 'Model', 'Fuel Consumption City (L/100km)']])

#c)

specific_size= data[(data['Engine Size (L)'] > 2.5) & (data['Engine Size (L)'] < 3.5)]
print('Average CO2 emissions for 2.5-3.5L vehicles:', specific_size['CO2 Emissions (g/km)'].mean())

#d)

audi = data[data['Make']=='Audi']

print(f"Number of audis:", len(audi))

audi_4cyl = data[(data['Make']=='Audi') & (data['Cylinders']==4)]

print(f"Average CO2 for 4 cyl audis:", audi_4cyl['CO2 Emissions (g/km)'].mean())

#e)

cylinder_counts = data['Cylinders'].value_counts().sort_index()
print(cylinder_counts)

cylinder_emissions = data.groupby('Cylinders')['CO2 Emissions (g/km)'].mean()
print("Cylinder emissions: ")
print(cylinder_emissions)

#f)

diesel = data[data['Fuel Type'] == 'D']
gas = data[data['Fuel Type'] == 'Z']

print(f'Average diesel consumption:', data[data['Fuel Type'] == 'D']['Fuel Consumption City (L/100km)'].mean())
print(f'Average diesel consumption median:', data[data['Fuel Type'] == 'D']['Fuel Consumption City (L/100km)'].median())
print(f'Average gas consumption:', data[data['Fuel Type'] == 'Z']['Fuel Consumption City (L/100km)'].mean())
print(f'Average gas consumption median:', data[data['Fuel Type'] == 'Z']['Fuel Consumption City (L/100km)'].median())

#g)

selection = data[(data['Cylinders'] == 4) & (data['Fuel Type'] == "D")].nlargest(1, 'Fuel Consumption City (L/100km)')

print("Najveca potrosnja 4 cilindra dizel:", selection)

#h)

manuals = data[(data['Transmission'].str[0] == 'M')]
length = len(manuals['Make'])
print(f"Postoji {length} vozila s rucnim mjenjacem")

#i)

print(data.corr(numeric_only=True))

'''
Komentiranje zadnjeg zadatka:
Velicine imaju dosta veliki korelaciju. Npr. broj obujam motora i broj cilindara su oko 0.9, dok je potrosnja oko 0.8 sto ukazuje na veliku korelaciju.
Takodjer razlog zasto potrosnja u mpg ima veliku negativnu korelaciju je to sto je ta velicina obrnuta, odnosno, sto automobil vise trosi, broj je manji
Sto je negativna korelacija blize -1 to je ona vise obrnuto proporcijalna, dok sto je blize 1, to je vise proporcijonalna. Vrijednosti oko 0
nemaju nikakvu korelaciju s velicinom.
'''