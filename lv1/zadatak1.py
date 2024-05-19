"""Zadatak 1.4.1 Napišite program koji od korisnika zahtijeva unos radnih sati te koliko je 
placen po radnom satu. Koristite ugradenu Python metodu ¯ input(). Nakon toga izracunajte 
koliko je korisnik zaradio i ispišite na ekran. Na kraju prepravite rješenje na nacin da ukupni 
iznos izracunavate u zasebnoj funkciji naziva ˇ total_euro.
Primjer:
Radni sati: 35 h
eura/h: 8.5
Ukupno: 297.5 eura"""

print("Unesi radne sate")
sati = float(input())    #input je string tipa pa moram pretvorit u float
print("Unesi satnicu")
satnica = float(input())

ukupno = sati * satnica

print(f"Radni sati: {sati} h\neura/h: {satnica}\nUkupno: {ukupno} eura")  #f ako hocu koristit varijable u stringu