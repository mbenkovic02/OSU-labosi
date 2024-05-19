"""Zadatak 1.4.3 Napišite program koji od korisnika zahtijeva unos brojeva u beskonacnoj petlji ˇ
sve dok korisnik ne upiše „Done“ (bez navodnika). Pri tome brojeve spremajte u listu. Nakon toga
potrebno je ispisati koliko brojeva je korisnik unio, njihovu srednju, minimalnu i maksimalnu
vrijednost. Sortirajte listu i ispišite je na ekran. Dodatno: osigurajte program od pogrešnog unosa
(npr. slovo umjesto brojke) na nacin da program zanemari taj unos i ispiše odgovaraju ˇ cu poruku."""

list = []

while True:
    a=input("Unesite broj")
    if a == "done" or a == "Done":
        break
    try:
        list.append(float(a))
    except:
        print("Unesite broj!")


print("Brojeva uneseno: " ,len(list))
print("Srednja vrijednost: ", sum(list)/len(list))
print("Minimalna vrijednost: ", min(list))
print("Maksimalna vrijednost: ", max(list))

list.sort
print("Sortirana lista: ", list)