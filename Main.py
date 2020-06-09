from Swarm import Swarm, np

Table_18 = [(21000, 50000), (50000, 84000), (786000, 987000), (97000, 156000), (125000, 239000), (1008000, 1383000),
            (902000, 115800), (462000, 96000), (465000, 790000), (545000, 908000), (311000, 396000), (675000, 984000),
            (128000, 189000), (105000, 103000), (215000, 285000), (31000, 70000), (42000, 90000), (78000, 73000), ]
border = int(len(Table_18) * 0.7)  # 70 % данных на тренировку ,30 % на проверку результата
train_18 = Table_18[:border]
test_18 = Table_18[border:]


def train(table_18=train_18):
    swarm_app_strat_1 = Swarm(min_x=0, max_x=10, n=2, population=300, generations=500, table=table_18, flag_test=1,
                              plt_name='strat_1')
    swarm_app_strat_2 = Swarm(min_x=0, max_x=10, n=2, population=300, generations=500, table=table_18, flag_test=2,
                              plt_name='strat_2')
    swarm_app_strat_1.start()
    swarm_app_strat_2.start()


def train2(table_18=train_18):
    swarm_app_strat_1 = Swarm(min_x=0, max_x=15, n=2, population=100, generations=500, table=table_18, flag_test=1,
                              plt_name='strat_3')
    swarm_app_strat_2 = Swarm(min_x=0, max_x=15, n=2, population=200, generations=500, table=table_18, flag_test=2,
                              plt_name='strat_4')
    swarm_app_strat_1.start()
    swarm_app_strat_2.start()


train()
train2()

def MD(x, T=train):
    s = 0.0
    for el in T:
        s += abs(x[0] * el[0] ** x[1] - el[1])
    return s

def test():
    List_of_coef = []
    with open('coef.txt', 'r') as f:
        for line in f:
            line = line.strip()
            List_of_coef.append(line.split(','))
    d = np.array(List_of_coef, dtype=np.float64)
