from lib import *

def main():
    from pandas import read_csv, DataFrame as df
    from sklearn import metrics as m
    # from numpy import array as na

    # 1
    x = read_csv('classification.csv')
    t = x['true']
    p = x['pred']

    # 2
    tp = len(x[(t == 1) & (p == 1)])
    fp = len(x[(t == 0) & (p == 1)])
    fn = len(x[(t == 1) & (p == 0)])
    tn = len(x[(t == 0) & (p == 0)])
    a = (tp, fp, fn, tn)
    pf('9_1', pp(a))
    print(sum(a) == len(x))

    # 3
    ma = m.accuracy_score(t, p)
    mp = m.precision_score(t, p)
    mr = m.recall_score(t, p)
    mf = m.f1_score(t, p)
    a = [round(e, 2) for e in (ma, mp, mr, mf)]
    pf('9_2', pp(a))

    # 4
    x = read_csv('scores.csv')
    # 5
    t = x['true']
    a = [m.roc_auc_score(t, x[[i]]) for i in range(1, 5)]
    pf('9_3', x.columns.values[a.index(max(a)) + 1])
    print(a)

    # 6
    def mt(i):
        a = df(list(m.precision_recall_curve(t, x[[i]]))).transpose()
        am = max(a[a[1] >= 0.7][0])
        print(x.columns.values[i], am)
        return am

    l = [mt(i) for i in range(1, 5)]
    pf('9_4', x.columns.values[l.index(max(l)) + 1])


start = time()
main()
finish(start)
