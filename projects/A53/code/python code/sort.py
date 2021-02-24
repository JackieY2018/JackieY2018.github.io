for day in range(1, 15):
    f = open("./data/day."+str(day), "r")
    line = f.readline()
    while line is not '':
        i = 0
        while line[i] != '\t':
            i += 1
        index = line[0:i:]
        inform = line[i + 1::]
        # if int(index) < 501:
        q = open("./data/sorted/"+index+".txt", "a")
        q.write(inform)
        q.close()
        line = f.readline()
    f.close()
    print(day, ' finished')
