f = open("./data/question", "r")
q = open("./data/keyword2/keyword.txt", 'r')
line = f.readline()
keys = q.readline()
while line is not '':
    i = 0
    while line[i] != '\t':
        i += 1
    index = line[0:i:]
    inform = line[i+1:-1:]
    key_str = keys[i+1:-2:]
    Question = list(map(int, inform.split()))
    key_lst = list(map(int, key_str.split()))
    new_key = key_lst.copy()
    for word in key_lst:
        if word not in Question:
            new_key.remove(word)
    n = open("./data/keyword2/newkeyword.txt", 'a')
    n.write(index + '\t')
    for j in range(min(len(new_key), 5)):
        n.write(str(new_key[j]) + ' ')
    n.write('\n')
    n.close()
    print(int(index), ' finished')
    line = f.readline()
    keys = q.readline()
f.close()
q.close()