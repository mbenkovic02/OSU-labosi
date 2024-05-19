def total_euro(hours, pay_rate):
    return hours * pay_rate

hours = input('Radni sati: ')
hours = hours.split(' ')[0]
hours = int(hours)

pay_rate = input('eura/h: ')
pay_rate = pay_rate.split(' ')[0]
pay_rate = float(pay_rate)

print(f"Ukupno: {total_euro(hours, pay_rate)} eura")

