from jieba import analyse

analyse.set_stop_words("./data/stopwords")

for i in range(1, 1001):
    f = open("./data/sorted/"+str(i)+".txt", 'r')
    data = f.read()
    keyword = analyse.extract_tags(data, topK=10)
    f.close()
    q = open("./data/keyword2/keyword.txt", 'a')
    q.write(str(i)+'\t')
    for key in keyword:
        q.write(key+' ')
    q.write('\n')
    q.close()
    print(i, ' finished')