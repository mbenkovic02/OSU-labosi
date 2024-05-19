"""Zadatak 1.4.2 Napišite program koji od korisnika zahtijeva upis jednog broja koji predstavlja
nekakvu ocjenu i nalazi se izmedu 0.0 i 1.0. Ispišite kojoj kategoriji pripada ocjena na temelju ¯
sljedecih uvjeta: ´
>= 0.9 A
>= 0.8 B
>= 0.7 C
>= 0.6 D
< 0.6 F
Ako korisnik nije utipkao broj, ispišite na ekran poruku o grešci (koristite try i except naredbe).
Takoder, ako je broj izvan intervala [0.0 i 1.0] potrebno je ispisati odgovaraju ¯ cu poruku."""


try:
    ocjena = float(input("Unesite ocjenu između 0.0 i 1.0: "))
    if ocjena < 0.0 or ocjena > 1.0:
     raise ValueError("Ocjena nije u intervalu")

    if ocjena >= 0.9:
     print("Ocjena: A")
    if ocjena >= 0.8:
     print("Ocjena: B")
    if ocjena >= 0.7:
     print("Ocjena: C")
    if ocjena >= 0.6:
     print("Ocjena: D")
    if ocjena < 0.6:
     print("Ocjena: F")

except ValueError as e:
    print("Greška:", e)