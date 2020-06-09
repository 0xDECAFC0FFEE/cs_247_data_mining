import collections
import random

def readlines(filename, encoding='utf-8', strip_newline=True):
    with open(filename, 'r', encoding=encoding) as fin:
        for line in fin:
            if strip_newline:
                yield line.rstrip('\n')
            else:
                yield line

def writelines(filename, content, encoding='utf-8', append_newline=True):
    with open(filename, 'w', encoding=encoding) as fout:
        for line in content:
            fout.write(line)

            if append_newline:
                fout.write('\n')


rec = {}

for phase in range(7):
    for line in readlines(f'./underexpose_test_qtime-{phase}.csv'):
        user_id, time = line.split(',')
        rec[user_id] = set()

for phase in range(7):
    for line in readlines(f'./underexpose_train_click-{phase}_ratio_u0_i0.test'):
        user_id, item_id, rating, time = line.split(',')
        if int(rating) and user_id in rec:
            rec[user_id].add(item_id)

for key in rec:
    lr = len(rec[key])
    if lr < 50:
        while len(rec[key]) < 50:
            rec[key].add(str(random.randint(0, 100000)))
    if lr > 50:
        rec[key] = set(random.sample(rec[key], 50))


writelines('submit.csv', [','.join([key, *rec[key]]) for key in rec])
