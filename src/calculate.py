def distance(x1, x2, y1, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

value = input('Enter the first point (x, y): ')
total = 0
count = 0

while value:
    value_list = [int(x) for x in value.split(',')]
    
    if count == 0:
        x1 = value_list[0]
        y1 = value_list[1]
    else:
        x2 = value_list[0]
        y2 = value_list[1]
        total += distance(x1, x2, y1, y2)
        x1 = x2  # Reset x1 to the new x2
        y1 = y2  # Reset y1 to the new y2
    
    value = input('Enter the next point (x, y) or press Enter to calculate: ')
    count += 1
64
scale = float(input('Enter the scale factor: '))
print('Total distance: ' + str(scale * total))
