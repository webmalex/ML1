from lib import *

def main():
    def pp(l): return ' '.join(map(str, l))

    # 1
    from pandas import read_csv
    x = read_csv('classification.csv')
    t = x['true']
    p = x['pred']

    # 2
    tp = len(x[(t == 1) & (p == 1)])
    fp = len(x[(t == 0) & (p == 1)])
    fn = len(x[(t == 1) & (p == 0)])
    tn = len(x[(t == 0) & (p == 0)])
    a1 = (tp, fp, fn, tn)
    pf('9_1', pp(a1))
    print(sum(a1) == len(x))

    # 3
    from sklearn import metrics as m
    ma = m.accuracy_score(t, p)
    mp = m.precision_score(t, p)
    mr = m.recall_score(t, p)
    mf = m.f1_score(t, p)
    a2 = [round(e, 2) for e in (ma, mp, mr, mf)]
    pf('9_2', pp(a2))

start = time()
main()
finish(start)
