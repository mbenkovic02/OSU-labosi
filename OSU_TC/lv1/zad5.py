def average_word_count(sms_list):
    total_words = 0
    for sms in sms_list:
        total_words += len(sms.split(" "))
    return total_words / len(sms_list)

def ends_with_ex_mark(sms):
    return sms[-1] == '!'

spam = []
ham = []

sms_file = open("lv1\SMSSpamCollection.txt")
for line in sms_file:
    line = line.rstrip()
    parts = line.split("\t")
    if (parts[0] == "ham"):
        ham.append(parts[1])
    elif (parts[0] == "spam"):
        spam.append(parts[1])



print(f"Average word count in spam {average_word_count(spam)}")
print(f"Average word count in ham {average_word_count(ham)}")
print(f"Number of spam ending with !: {len(list(filter(ends_with_ex_mark, spam)))}")

sms_file.close()
