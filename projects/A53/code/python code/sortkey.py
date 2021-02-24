import xlsxwriter


def cal_rel(key1, key2):
    l1 = set(key2q[key1])
    l2 = set(key2q[key2])
    s = l1 & l2
    return len(s)


key2q = {}
key_lst = []
f = open("./data/keyword2/newkeyword.txt", "r")
line = f.readline()
while line is not '':
    i = 0
    while line[i] != '\t':
        i += 1
    index = int(line[0:i:])
    keys = line[i+1:-2:]
    keywords = list(map(int, keys.split()))
    for key in keywords:
        if key in key2q:
            key2q[key].append(index)
        else:
            key2q[key] = [index]
            key_lst.append(key)
    line = f.readline()
f.close()
lst = key_lst.copy()
for key in lst:
    if len(key2q[key]) <= 10:
        del key2q[key]
        key_lst.remove(key)
Question = [i for i in range(1, 1001)]
for lst in key2q.values():
    for q in lst:
        if q in Question:
            Question.remove(q)
print(Question)

workbook = xlsxwriter.Workbook('relationofkeyword.xlsx')
worksheet = workbook.add_worksheet()

for i in range(len(key_lst)):
    worksheet.write(i+1, 0, key_lst[i])
    worksheet.write(0, i+1, key_lst[i])
    worksheet.write(i+1, i+1, '\\')

for i in range(len(key_lst)):
    for j in range(i+1, len(key_lst)):
        r = cal_rel(key_lst[i], key_lst[j])
        worksheet.write(i+1, j+1, r)
        worksheet.write(j+1, i+1, r)

workbook.close()

for x in range(3,10):
    for y in range(2, 5):
        relation_lst = []
        for i in range(len(key_lst)):
            for j in range(i + 1, len(key_lst)):
                if cal_rel(key_lst[i], key_lst[j]) >= x:
                    s = set()
                    s.add(key_lst[i])
                    s.add(key_lst[j])
                    relation_lst.append(s)
        for i in range(5):
            lst = relation_lst.copy()
            for s1 in lst:
                for s2 in lst:
                    if s1 != s2 and s1.isdisjoint(s2) is False and s1 | s2 not in relation_lst:
                        s3 = (s1 | s2) - (s1 & s2)
                        if s3 in relation_lst:
                            relation_lst.append(s1 | s2)

        lst = relation_lst.copy()
        for s1 in lst:
            if s1 in relation_lst and len(s1) <= y:
                relation_lst.remove(s1)
            else:
                for s2 in lst:
                    if s1 in relation_lst and s1 != s2 and s1 <= s2:
                        relation_lst.remove(s1)

        print(relation_lst)
        print(len(relation_lst))
        Question2 = [i for i in range(1, 1001)]
        for s in relation_lst:
            for key in s:
                for q in key2q[key]:
                    if q in Question2:
                        Question2.remove(q)
        print(len(Question2))
        print(Question2)
        print(x, '\t', y, '\t', len(relation_lst), '\t', len(Question2))
print(key2q)


