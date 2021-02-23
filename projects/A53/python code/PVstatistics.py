import xlsxwriter


workbook = xlsxwriter.Workbook('result0.xlsx')
worksheet = workbook.add_worksheet()
for index in range(1000):
    worksheet.write(0, index, index + 1)
for day in range(1, 15):
    result_lst = [0 for index in range(1001)]
    f = open("./data/day."+str(day), "r")
    line = f.readline()
    while line is not '':
        i = 0
        s = ''
        while line[i] != '\t':
            s += line[i]
            i += 1
        result_lst[int(s)] += 1
        line = f.readline()
    print(result_lst)
    f.close()
    for index in range(1000):
        worksheet.write(day, index, result_lst[index + 1])
workbook.close()
