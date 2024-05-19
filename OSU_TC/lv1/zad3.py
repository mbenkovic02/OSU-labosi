numbers = []
while(True):
    try:
        num = input('Enter a number: ')
        if num == 'Done':
            break
        numbers.append(float(num))
    except:
        print('NOT A NUMBER!')

print(f"Numbers entered: {len(numbers)}")
print(f"Average value: {sum(numbers) / len(numbers)}")
print(f"Minimum: {min(numbers)}")
print(f"Maximum: {max(numbers)}")
