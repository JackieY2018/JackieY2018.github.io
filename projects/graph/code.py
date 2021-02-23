import json
import matplotlib.pyplot as plt


class Queue:  # 队列
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0, item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)


class Vertex:  # 顶点
    def __init__(self, actor):
        self.id = actor  # id:str,为该顶点对应演员名字的字符串
        self.films = []  # films:list,list中的元素均为包含电影信息的dict
        self.nbrs = {}  # nbr:dict,其key为与自己相连接的Vertex，value为对应的Edge
        self.color = 'white'  # color:str,初始为'white',可能存在的状态还有'gray'和'black'
        self.dist = 0  # dist:int,代表在bfs中距离起始顶点的距离

    def getId(self):  # 返回值为str
        return self.id

    def getFilms(self):  # 返回值为list
        return self.films

    def getNbr(self):  # 返回值为dict
        return self.nbrs

    def getDistance(self):  # 返回值为int
        return self.dist

    def getColor(self):  # 返回值为str
        return self.color

    def addFilm(self, film):  # 将包含电影信息的dict加入到films列表中
        self.films.append(film)

    def addNbr(self, nbr, edge):  # nbr:Vertex, edge:Edge, 与nbr相连并将边设置为edge
        self.nbrs[nbr] = edge

    def setColor(self, color):  # color:'white','gray'或'black'中的一个
        self.color = color

    def setDistance(self, d):  # d:int
        self.dist = d

    def __str__(self):  # 用于检查顶点的内容，输出这个顶点的演员的名字、他所演电影的名称、与他相连接的演员的名字
        s = str(self.id) + ":\nFilms:"
        for film in self.films:
            s1 = film['title']
            s += s1
            s += ";"
        s += "\nNbrs:"
        for nbr in self.nbrs:
            s1 = nbr.getId()
            s += s1
            s += ";"
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


class Edge:  # 为边单独创建了一个类
    def __init__(self, v1, v2, film_lst=[]):
        self.endpoints = {v1, v2}  # endpoint:set, 其中v1,v2均为Vertex, 为边连接的节点
        self.films = film_lst  # films:list,list中的元素均为包含电影信息的dict
        self.filmnum = len(film_lst)  # filmnum:int, 为这条边上的电影数目

    def addFilm(self, film):  # 将一个电影添加到这个边上
        self.films.append(film)
        self.filmnum += 1

    def getEndpt(self):  # 获取这个边所连接的两个Vertex, 返回值为set
        return self.endpoints

    def getFilms(self):  # 返回值为list
        return self.films


class Graph:
    def __init__(self):
        self.vertices = {}  # vertices:dict. key:str,为演员名字; value: Vertex, 为对应的顶点
        self.numVertices = 0  # numVertices:int, 为Graph中顶点的个数

    def getVertex(self, actor):  # actor:str,为一个演员名字, 返回值为图中该演员对应的顶点，如果图中没有这个演员返回值为None
        if actor in self.vertices:
            return self.vertices[actor]
        else:
            return None

    def getAllVertex(self):  # 返回值为一个包含所有顶点的list
        return list(self.vertices.values())

    def getActors(self):  # 返回值为一个包含所有演员名字的list
        return list(self.vertices.keys())

    def getEdge(self, actor1, actor2):  # actor1:str, actor2:str,返回值为两个演员之间的边Edge,如果没有边则返回None
        v1 = self.vertices[actor1]
        v2 = self.vertices[actor2]
        nbr1 = v1.getNbr()
        if v2 in nbr1:
            return nbr1[v2]
        return None

    def addVertex(self, actor):  # actor:str, 将名字为actor的演员对应的顶点添加到图中
        self.numVertices = self.numVertices + 1
        self.vertices[actor] = Vertex(actor)

    def addFilm_V(self, actor, film):  # actor:str, film:dict, 向演员actor对应的顶点中添加一个电影，电影格式为储存了电影信息的dict
        self.vertices[actor].addFilm(film)

    def addEdge(self, actor1, actor2, film=None):  # actor1:str, actor2:str, film:dict, 在actor1和actor2对应的顶点之间添加一条边
        # film为actor1和actor2共同出演的电影, 给这条边也附加这个共同出演的电影
        if actor1 != actor2:  # 为了避免出现自环
            newEdge = Edge(self.vertices[actor1], self.vertices[actor2])
            if film:
                newEdge.addFilm(film)
            self.vertices[actor1].addNbr(self.vertices[actor2], newEdge)
            self.vertices[actor2].addNbr(self.vertices[actor1], newEdge)

    def addFilm_E(self, actor1, actor2, film):  # actor1:str, actor2:str, film:dict, 向actor1和actor2对应的顶点之间的边上添加一个电影
        edge = self.getEdge(actor1, actor2)
        if edge:
            edge.addFilm(film)
        else:  # 如果actor1和actor2对应的顶点之间没有边，则添加一条边
            self.addEdge(actor1, actor2, film)

    def __contains__(self, actor):  # actor:str, 判断名字为actor的演员是否在图中
        return actor in self.vertices

    def __iter__(self):  # 迭代器，返回图中的Vertex
        return iter(self.vertices.values())


def buildGraph():  # 用于建图, 返回值为一个Graph
    f = open('Film.json', 'r')
    allfilm = json.load(f)
    f.close()
    graph = Graph()
    for film in allfilm:  # 对每个电影进行逐一处理
        actors = film['actor'].split(',')
        for actor in actors:  # 先将电影的演员创建Vertex加入到图中
            if actor not in graph:
                graph.addVertex(actor)
            graph.addFilm_V(actor, film)  # 将这部电影添加到每个演员对应的Vertex中
        num = len(actors)
        if num >= 2:  # 演员不少于2人时在演员之间两两构建一条边
            for i in range(num - 1):
                for j in range(i + 1, num):
                    # 在addFilm_E函数中，如果actors[i]和actors[j]之间没有边则会先建立边，如果有边则会直接在边中添加电影film
                    graph.addFilm_E(actors[i], actors[j], film)
    return graph


def bfs(start, Compnt=False, Dist=False, ColorReverse=False):
    # start:Vertex, 为进行广度优先搜索的起始顶点
    # Compnt:bool, 为True时返回值为set, 为广度优先搜索搜索到的顶点的集合
    # Dist:bool, 为True时返回值为int, 为广度优先搜索能搜到的顶点中与起始点的最远距离,
    # ColorReverse:bool, 广度优先搜索进行第一次后，会将能搜到的节点的color全部变为'black',
    # 因此ColorReverse为True的时候规定起始颜色为'black', 搜索中为'gray', 搜索完成后为'white', 即黑白互换
    currentComponent = set()
    d = 0
    white = 'black' if ColorReverse else 'white'  # 这里的white代表bfs之前起始节点和能搜到的节点的color
    black = 'white' if ColorReverse else 'black'  # 这里的black代表bfs之后起始节点和能搜到的节点的color
    start.setDistance(0)
    vertexQueue = Queue()  # bfs过程中借助队列这一数据结构
    vertexQueue.enqueue(start)
    while not vertexQueue.isEmpty():
        current = vertexQueue.dequeue()
        for nbr in current.getNbr():
            if nbr.getColor() == white:  # 未搜索时color为white
                nbr.setColor('gray')   # 搜索中color为'gray'
                nbr.setDistance(current.getDistance() + 1)
                vertexQueue.enqueue(nbr)
        current.setColor(black)  # 搜索完成后color为black
        if Compnt:
            currentComponent.add(current)  # 将搜索到的顶点加入集合
        if Dist and d < current.getDistance():
            d = current.getDistance()  # 将最远的dist记录下来
    if Compnt:
        return currentComponent
    if Dist:
        return d


def getComponents(graph):  # 将图中所有联通分支输出
    # 返回值components为一个list，其中元素为set, 每一个set为一个联通分支中的顶点的集合
    # components的元素按照各自集合的大小降序排列
    components = []
    restVertex = set(graph.getAllVertex())
    # 因为一次bfs所能搜索到的顶点就构成一个连通分支，所以将还没搜索到的节点记录在集合中，可以避免在获取连通分支时的重复搜索
    while len(restVertex) != 0:
        curComponent = bfs(restVertex.pop(), Compnt=True)  # 为一个得到的连通分支
        components.append(curComponent)
        # print('getComponents:', len(components), ' finished')
        # 用于监视获取连通分支的进程
        restVertex = restVertex - curComponent  # 利用差集将已得到连通分支中的顶点从剩余顶点中剔除
    components.sort(key=lambda x: len(x), reverse=True)   # 根据连通分支的规模降序排列
    # components.sort(key=lambda x: (len(x), len(getEdges(x))), reverse=True)
    # 顶点数相同时按照边数排序, 但实际运行时会大大降低效率，并且该二级排序意义并不显著
    return components


def getFilms(component):  # component:set, 集合中的元素为Vertex
    # 返回值film_dic:dict, key:str为电影的id, value:dict为电影信息对应的集合
    # 返回的film_dic包含集合中所有演员出演的电影
    film_dic = {}
    for vertex in component:
        films = vertex.getFilms()
        for film in films:
            film_dic[film['_id']['$oid']] = film
            # 使用字典可以很好地避免因为电影重复出现带来的麻烦，虽然set也可以实现，但是film为dict, 是unhashable类, 因此不能使用set
    return film_dic


def getType(film_dic):  # film_dic:dict, 与getFilms返回值的数据类型相同
    # 返回值为list, 为film_dic中所有电影类别排行前3, 如果不足3种类别则全部输出
    types = {}  # key:str为电影类别, value:int为该类别出现的次数
    for film in film_dic.values():
        type_lst = film['type'].split(',')
        for t in type_lst:
            if t in types:
                types[t] += 1
            else:
                types[t] = 1
    result = list(types.keys()).copy()
    result.sort(key=lambda x: (types[x], x), reverse=True)  # 在类别出现次数相同时按字符串大小排序
    return result[:3:]


def getAveStar(film_dic):  # film_dic:dict, 与getFilms返回值的数据类型相同
    # 返回值为float, 为film_dic中所有电影的平均星级数，保留到小数点后2位
    total_star = 0
    for film in film_dic.values():
        total_star += film['star']
    return round(total_star/float(len(film_dic)), 2)


def getDiameter(component):  # component:set, 集合中的元素为Vertex, 为一个连通分支
    # 返回值为int, 为该连通分支的直径
    diameter = 0
    for vertex in component:  # 对每个顶点都作为起始点进行bfs, 返回最大距离 这些最大距离的最大值为该连通分支的直径
        reverse = (vertex.getColor() == 'black')  # 用于判断联通分支内是否均为黑色，如果是则需要在bfs时黑白互换
        d = bfs(vertex, Compnt=False, Dist=True, ColorReverse=reverse)
        if d > diameter:
            diameter = d
    return diameter


def getPartners(graph, actor):  # graph:Graph, actor:str
    # 返回值为set, 集合中元素均为Vertex, 且为演员actor和actor曾经的共同出演者们构成的集合
    vertex = graph.getVertex(actor)
    partners = {vertex}
    for partner in vertex.getNbr():
        partners.add(partner)
    return partners


def draw_bar(scale, diameter, star):  # scale, diameter, star均为list, 该函数用于将这三组数据绘制成柱形图
    label_list = [i for i in range(1, 21)]
    label_list.extend([i for i in range(4558, 4578)])
    # label_list为柱形图的横轴标签, 代表连通分支按规模排名的名次
    x = [i for i in range(40)]
    fig, axs = plt.subplots(4, 1, sharex=True, gridspec_kw={'height_ratios': [1, 3, 4, 4]})
    # 将三个柱形图作为子图画在一张图中, 并共用x轴, 同时对scale柱形图进行纵坐标截断的处理
    # 先绘制含有纵坐标截断的scale柱形图
    axs[1].bar(x, scale, color='royalblue')
    axs[0].bar(x, scale, color='royalblue', label='Scale')
    axs[1].set_ylim(0, 54)
    axs[0].set_ylim(84680, 84690)
    axs[1].spines['top'].set_visible(False)
    axs[0].spines['bottom'].set_visible(False)
    axs[1].xaxis.tick_bottom()
    axs[0].xaxis.tick_top()
    plt.subplots_adjust(hspace=0.04)
    # 为截断柱形图的纵坐标架绘制截断双斜线://
    d = 0.01
    kwargs = dict(transform=axs[0].transAxes, color='k', clip_on=False)
    axs[0].plot((-d, +d), (-3 * d, +3 * d), **kwargs)
    axs[0].plot((1 - d, 1 + d), (-3 * d, +3 * d), **kwargs)
    kwargs.update(transform=axs[1].transAxes)
    axs[1].plot((-d, +d), (1 - d, 1 + d), **kwargs)
    axs[1].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    # 绘制diameter和star的柱形图
    axs[2].bar(x, diameter, color='seagreen', label='Diameter')
    axs[3].bar(x, star, tick_label=label_list, color='darkorange', label='star')
    axs[2].set_ylim(-1.2, 5.5)
    axs[3].set_ylim(0, 10.9)
    # 生成图例
    axs[0].legend()
    axs[2].legend()
    axs[3].legend()
    # 设置横纵坐标的名称
    axs[3].set_xlabel("Rank")
    axs[1].set_ylabel("Scale")
    axs[2].set_ylabel("Diameter")
    axs[3].set_ylabel("Ave. Stars")
    # 将横坐标刻度的标签自己转90度
    for tick in axs[3].get_xticklabels():
        tick.set_rotation(90)
    # 在柱形图的柱子上方添加该柱子对应的具体值
    for i, y in zip(x, scale):
        axs[0].text(i, y - 0.02, y, ha='center', va='bottom')
    for i, y in zip(x, scale):
        axs[1].text(i, y - 0.02, y, ha='center', va='bottom')
    for i, y in zip(x, star):
        axs[3].text(i, y - 0.02, y, ha='center', va='bottom')
    for i, y in zip(x, diameter):
        if y < 0:
            axs[2].text(i, 0.01, y, ha='center', va='bottom')
        elif y >= 0:
            axs[2].text(i, y - 0.02, y, ha='center', va='bottom')
    # 输出结果, 完成绘制
    plt.savefig('result.png')
    plt.show()


def getEdges(component):  # component:set, 集合中的元素为Vertex, 为一个连通分支
    # 返回值为int, 为该连通分支中的所有边的集合
    total_edges = set()
    for vertex in component:
        edges = set(vertex.getNbr().values())
        total_edges = total_edges | edges
    return total_edges


def prim(start):  # start:Vertex, 为Prim算法的起始顶点
    # 因为这里的图中的边权重均为1, 所以Prim算法可以极大地简化
    # 对这种特殊情形，只需要队列即可，无需优先队列
    # 返回值为set, 为最小生成树中包括的边的集合
    vertex_q = Queue()
    min_edges = set()
    if start.getColor() == 'white':
        white = 'white'
        black = 'black'
    else:
        white = 'black'
        black = 'white'
    vertex_q.enqueue(start)
    start.setColor(black)
    while not vertex_q.isEmpty():
        current = vertex_q.dequeue()
        nbrs = current.getNbr()
        for nbr in nbrs:
            if nbr.getColor() == white:
                vertex_q.enqueue(nbr)
                nbr.setColor(black)
                min_edges.add(nbrs[nbr])
    return min_edges


def main():
    g = buildGraph()  # 建立图, g为建好Graph
    components = getComponents(g)  # components为g中所有连通分支的list, list的每个元素为一个连通分支组成的set
    print('连通分支总个数:', len(components))
    bottoms = components[-20::]  # 连通分支中规模最小的20个
    top_bottom = components[:20:]  # 规模最大的20个
    top_bottom.extend(bottoms)  # 将规模最小的20个接到规模最大的20个之后
    scale = []
    star = []
    types = []
    diameter = []
    for i in range(len(top_bottom)):  # 依次处理这40个连通分支
        scale.append(len(top_bottom[i]))
        films = getFilms(top_bottom[i])
        types.append(getType(films))
        star.append(getAveStar(films))
        if i != 0:
            diameter.append(getDiameter(top_bottom[i]))
        else:  # 最大的连通分支直径设为-1
            diameter.append(-1)
    draw_bar(scale, diameter, star)  # 将结果绘制成柱形图
    print('scale:', scale)
    print('types:', types)
    print('star:', star)
    print('diameter:', diameter)
    vertex_zxc = g.getVertex('周星驰')  # 获得周星驰对应的顶点
    zxc = {vertex_zxc}
    print('周星驰的电影的平均星级数：', getAveStar(getFilms(zxc)))
    print('周星驰和共同出演者：')
    z_partners = getPartners(g, '周星驰')  # 为周星驰和共同出演者构成的顶点集合
    z_films = getFilms(z_partners)  # 为周星驰和共同出演者一共演过的电影构成的dict
    print("总人数：", len(z_partners))
    print("总电影数：", len(z_films))
    print("所演电影平均星级数：", getAveStar(z_films))
    print("所演电影类别前三名：", getType(z_films))
    print('对规模前20的连通分支的最小生成树的探究：')
    # 并没有计算最大规模的连通分支，因为对2-19名连通分支的研究已经足够得出结论
    total_edge_nums = []  # 该列表中的元素为连通分支的总边数
    min_edge_nums = []
    # 该列表中的元素为集合，每个集合对应这个连通分支以不同顶点为起点得到的最小生成树的边数的集合
    # 使用集合是为了探究最小生成树的边数是否与顶点的选取有关
    for i in range(1, 20):
        total_edge_nums.append(len(getEdges(top_bottom[i])))
        edgenums = set()
        for vertex in top_bottom[i]:
            edgenums.add(len(prim(vertex)))
        min_edge_nums.append(edgenums)
    print('总边数：', total_edge_nums)
    print('最小生成树中边数：', min_edge_nums)


main()
