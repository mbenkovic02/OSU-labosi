word_count = {}

song_file = open("lv1\song.txt")
for line in song_file:
    line = line.rstrip()
    words = line.split(" ")
    for word in words:
        if word not in word_count:
            word_count[word] = 1
            continue
        word_count[word] = word_count[word] + 1
song_file.close()
unique_words = 0
for word in word_count:
    if word_count[word] == 1:
        unique_words += 1
        print(f"{word} : {word_count[word]}")
print(unique_words)

