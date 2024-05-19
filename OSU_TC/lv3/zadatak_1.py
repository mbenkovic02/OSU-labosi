import pandas as pd 

data = pd.read_csv('lv3\data_C02_emission.csv')

# Zadatak pod a)
length = len(data['Make'])
print(f'DataFrame ima {length} mjerenja')

for col in data.columns:
    print(f"{col} has a type of {data[col].dtype}")

data['Vehicle Class'] = data['Vehicle Class'].astype('category')

print(f"Redovi s izostalim vrijednostima: {data.isnull().sum()}")
print(f"Duplicirane vrijednosti: {data.duplicated().sum()}")

# Zadatak pod b)
least_consuming = data.nsmallest(3, 'Fuel Consumption City (L/100km)')
most_consuming = data.nlargest(3, 'Fuel Consumption City (L/100km)')

print('Most consuming: ')
print(most_consuming[['Make', 'Model', 'Fuel Consumption City (L/100km)']])
print('Least consuming: ')
print(least_consuming[['Make', 'Model', 'Fuel Consumption City (L/100km)']])

# Zadatak pod c) 
selected_data = data[(data['Engine Size (L)'] >= 2.5) & (data['Engine Size (L)'] <= 3.5)]
length = len(selected_data['Make'])
print(f"Postoji {length} vozila koje imaju motor izmedu 2.5 i 3.5 L")

print(f"Prosjecni C02 ovih vozila jest: {selected_data['CO2 Emissions (g/km)'].mean()} g/km")

# Zadatak pod d)
selected_data = data[(data['Make'] == 'Audi')]
length = len(selected_data['Make'])
print(f"U mjerenjima ima {length} mjerenja koja se odnose na marku Audi")

selected_data = selected_data[(selected_data['Cylinders'] == 4)]
print(f"Prosjecni CO2 4 cilindrasa marke Audi je {selected_data['CO2 Emissions (g/km)'].mean()} g/km")

# Zadatak pod e)
cylinder_counts = data['Cylinders'].value_counts().sort_index()
print(cylinder_counts)

cylinder_emissions = data.groupby('Cylinders')['CO2 Emissions (g/km)'].mean()
print("Cylinder emissions: ")
print(cylinder_emissions)

# Zadatak pod f)
diesels = data[(data['Fuel Type'] == 'D')]
petrols = data[(data['Fuel Type'] == 'Z')]

print(f"Dizeli:\nProsjecno: {diesels['Fuel Consumption City (L/100km)'].mean()} - Medijalno: {diesels['Fuel Consumption City (L/100km)'].median()}")
print(f"Benzinci:\nProsjecno: {petrols['Fuel Consumption City (L/100km)'].mean()} - Medijalno: {petrols['Fuel Consumption City (L/100km)'].median()}")

# Zadatak pod g)
four_cylinder_diesels = diesels[(diesels['Cylinders'] == 4)]
print(f"4 cilindricni dizel koji najvise goriva trosi u gradu jest:\n{four_cylinder_diesels.nlargest(1, 'Fuel Consumption City (L/100km)')}")

# Zadatak pod h)
manuals = data[(data['Transmission'].str[0] == 'M')]
length = len(manuals['Make'])
print(f"Postoji {length} vozila s rucnim mjenjacem")

#Zadatak pod i)
print(data.corr(numeric_only=True))

'''
Komentiranje zadnjeg zadatka:
Velicine imaju dosta veliki korelaciju. Npr. broj obujam motora i broj cilindara su oko 0.9, dok je potrosnja oko 0.8 sto ukazuje na veliku korelaciju.
Takodjer razlog zasto potrosnja u mpg ima veliku negativnu korelaciju je to sto je ta velicina obrnuta, odnosno, sto automobil vise trosi, broj je manji
Npr: automobil koji trosi 25 MPG trosi vise nego automobil koji trosi 45 MPG. Dakle, ta velicina je obrnuta L/100km te takodjer, zbog toga dobivamo negativnu
korelaciju. Sto je negativna korelacija blize -1 to je ona vise obrnuto proporcijalna, dok sto je blize 1, to je vise proporcijonalna. Vrijednosti oko 0
nemaju nikakvu korelaciju s velicinom.
'''
