"""Zadatak 1.4.5 Napišite Python skriptu koja ce u ´ citati tekstualnu datoteku naziva ˇ SMSSpamCollection.txt
[1]. Ova datoteka sadrži 5574 SMS poruka pri cemu su neke ozna ˇ cene kao ˇ spam, a neke kao ham.
Primjer dijela datoteke:
ham Yup next stop.
ham Ok lar... Joking wif u oni...
spam Did you hear about the new "Divorce Barbie"? It comes with all of Ken’s stuff!
12 Poglavlje 1. Uvod u programski jezik Python
a) Izracunajte koliki je prosje ˇ can broj rije ˇ ci u SMS porukama koje su tipa ham, a koliko je ˇ
prosjecan broj rije ˇ ci u porukama koje su tipa spam. ˇ
b) Koliko SMS poruka koje su tipa spam završava usklicnikom ?"""


ham_messages=0
spam_messages=0
ham_words=0
spam_words=0
spam_exclamation_sentences=0

fhand = open ("lv1\SMSSpamCollection.txt")   #returns file object
lines = fhand.readlines()  #returns list of strings

for line in lines :
    
    label, message = line.split("\t")
    words = message.split()
    num_words = len(words)
    
    if(label== "ham"):
        ham_messages+=1
        ham_words+=num_words
    elif (label== "spam"):
        spam_messages+=1
        spam_words+=num_words
        
        if message.strip().endswith("!"):
            spam_exclamation_sentences += 1


average_words_ham = ham_words / ham_messages
average_words_spam = spam_words / spam_messages

print("Average number of words in ham messages:", average_words_ham)
print("Average number of words in spam messages:", average_words_spam)
print("Number of spam messages ending with an exclamation mark:", spam_exclamation_sentences)