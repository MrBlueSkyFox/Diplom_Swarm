from Swarm import Swarm, np, plt
from operator import itemgetter


def NS_data_to_table(data_file_name):
    list_of_data = []
    with open(data_file_name, 'r') as f:
        for line in f:
            line = line.strip()
            list_of_data.append(line.split(' '))
    border = int(len(list_of_data) * 0.7)  # 70 % данных на тренировку ,30 % на проверку результата
    data_table = np.array(list_of_data, dtype=np.float64)
    train_table = data_table[:border]
    test_table = data_table[border:]
    return train_table, test_table


train_18, test_18 = NS_data_to_table("NASA_18.txt")
train_60, test_60 = NS_data_to_table("NASA_60.txt")


def train_18_func(table=train_18, max_x=15, pop=300, gen=500):
    swarm_app_start_md = Swarm(min_x=0, max_x=max_x, n=2, population=pop, generations=gen, table=table, flag_choice=1,
                               plt_name='strat_18_md')
    swarm_app_start_md.start()
    swarm_app_start_mmre = Swarm(min_x=0, max_x=max_x, n=2, population=pop, generations=gen, table=table, flag_choice=2,
                                 plt_name='strat_18_mmre')
    swarm_app_start_mmre.start()
    swarm_app_start_rms = Swarm(min_x=0, max_x=max_x, n=2, population=pop, generations=gen, table=table, flag_choice=3,
                                plt_name='strat_18_rms')
    swarm_app_start_rms.start()
    swarm_app_start_ed = Swarm(min_x=0, max_x=max_x, n=2, population=pop, generations=gen, table=table, flag_choice=4,
                               plt_name='strat_18_ed')
    swarm_app_start_ed.start()


def train2(table=train_18, max_x=15, pop=300, gen=500):
    swarm_app_strat_1 = Swarm(min_x=0, max_x=max_x, n=2, population=pop, generations=gen, table=table, flag_test=1,
                              plt_name='strat_3')
    swarm_app_strat_2 = Swarm(min_x=0, max_x=max_x, n=2, population=pop, generations=gen, table=table, flag_test=2,
                              plt_name='strat_4')
    swarm_app_strat_1.start()
    swarm_app_strat_2.start()


def train_60_func(table=train_60, max_x=15, pop=300, gen=500):
    swarm_app_start_md = Swarm(min_x=0, max_x=max_x, n=2, population=pop, generations=gen, table=table, flag_choice=1,
                               plt_name='strat_60_md')
    swarm_app_start_md.start()
    swarm_app_start_mmre = Swarm(min_x=0, max_x=max_x, n=2, population=pop, generations=gen, table=table, flag_choice=2,
                                 plt_name='strat_60_mmre')
    swarm_app_start_mmre.start()
    swarm_app_start_rms = Swarm(min_x=0, max_x=max_x, n=2, population=pop, generations=gen, table=table, flag_choice=3,
                                plt_name='strat_60_rms')
    swarm_app_start_rms.start()
    swarm_app_start_ed = Swarm(min_x=0, max_x=max_x, n=2, population=pop, generations=gen, table=table, flag_choice=4,
                               plt_name='strat_60_ed')
    swarm_app_start_ed.start()


def MD(x, T=train_18):
    s = 0.0
    for el in T:
        s += abs(x[0] * el[0] ** x[1] - el[1])
    return s


def ef_check(x, table):
    s = []
    for el in table:
        est_ef = x[0] * el[0] ** x[1]
        dif = abs(est_ef - el[1])
        reply = 'Est ef= ' + str(est_ef) + ' ' + ' Real ef= ' + str(el[1]) + ' Dif= ' + str(dif)
        s.append(reply)
    with open('check.txt', 'a')as f:
        f.write('\n'.join(map(str, s)) + '\n')
        f.write('-------\n')


def plot_for_coef(x, table, name):
    predict_cocomo = []
    write_for_table = []  # вывод значения ef predict, Ef Nasa
    for el in table:
        est_ef = x[0] * el[0] ** x[1]
        predict_cocomo.append([el[0], est_ef])
        write_for_table.append([el[1], est_ef])
    with open('write_test_ef.txt', 'a')as f:
        f.write(' '.join(map(str, write_for_table)) + '\n')
        f.write('-------\n')
    # x1_y1 = list(zip(*table))
    # x2_y2 = list(zip(*predict_cocomo))
    x1_y1 = list(zip(*table))
    x2_y2 = list(zip(*predict_cocomo))
    fig, ax = plt.subplots()
    ax.set_title('a= ' + str(x[0]) + ' b= ' + str(x[1]))
    ax.plot(*x1_y1, '--ob', label='NASA data')
    ax.plot(*x2_y2, '--vr', label='Comp Val')
    # ax.axis('equal')
    leg = ax.legend();
    plt.savefig(name)
    # plt.show()
    plt.close(fig)


def test_many(tabl, full_table, name):
    List_of_coef = []
    with open('coef.txt', 'r') as f:
        for line in f:
            line = line.strip()
            List_of_coef.append(line.split(','))
    a_b_array = np.array(List_of_coef, dtype=np.float64)
    i = 0
    s_f = sorted(full_table, key=itemgetter(0))
    full_table = np.array(s_f)
    for a_b in a_b_array:
        ef_check(a_b, tabl)
        plot_for_coef(a_b, full_table, name + "_" + str(i))
        i = i + 1


train_18_func(train_18, 10, 400, 1000)
train_60_func(train_60, 10, 400, 1000)
# train2(train_18, 10, 400, 1000)

# test_many(test_18)
table18 = np.concatenate((test_18, train_18))
# test_many(test_18, table18, 'Est_Ef_VS_Ef')
