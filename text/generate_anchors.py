a = 9
anchors = []
for i in range(9):
    a = a / 0.7
    anchors.append('15,{}'.format(int(a)))
print(', '.join(anchors))

with open('text/icdar_2019_art.txt') as f:
    lines = f.readlines()
    for line in lines:
        for box in line.split(' ')[1:]:
            box = box.split(',')
            for i in range(len(box)):
                box[i] = int(box[i])
            if box[0] < 0 or box[1] < 0 or box[2] < 0 or box[3] < 0:
                print('smaller:{}'.format(line.split(' ')[0]))
            elif box[0] > 607 or box[1] > 607 or box[3] > 607 or box[2] > 607:
                print('bigger:{}'.format(line.split(' ')[0]))
