from builtins import print
import time


# import timeit

def t1():
    import numpy as np
    X = np.random.normal(loc=1, scale=10, size=(1000, 50))
    print(X)

    m = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_norm = ((X - m) / std)
    print(X_norm)

    Z = np.array([[4, 5, 0],
                  [1, 9, 3],
                  [5, 1, 1],
                  [3, 3, 3],
                  [9, 9, 9],
                  [4, 7, 1]])
    r = np.sum(Z, axis=1)
    print(np.nonzero(r > 10))

    A = np.eye(3)
    B = np.eye(3)
    print(A)
    print(B)

    AB = np.vstack((A, B))
    print(AB)


def pf(name, value):
    f = open(name + '.txt', 'w')
    f.write(str(value))
    f.close()
    print(name + '="%s"\n' % value)


def lesson1():
    import pandas
    data = pandas.read_csv('train.csv', index_col='PassengerId')

    def q1_1():
        d = data['Sex'].value_counts()
        pf('1_1', '%s %s' % (d[0], d[1]))
        print(d, '\n')

    def q1_2():
        d = data['Survived'].value_counts()
        r = round(d[1] / d.sum() * 100, 2)
        pf('1_2', r)
        print(d, '\n')

    def q1_3():
        d = data['Pclass'].value_counts()
        r = round(d[1] / d.sum() * 100, 2)
        pf('1_3', r)
        print(d, '\n')

    def q1_4():
        d = data['Age']
        a = round(d.sum() / d.count(), 2)
        m = d.median()
        pf('1_4', '%s %s' % (a, m))

    def q1_5():
        d = round(data['SibSp'].corr(data['Parch']), 2)
        pf('1_5', d)

    q1_1()
    q1_2()
    q1_3()
    q1_4()
    q1_5()


def lesson2():
    def t1():
        from sklearn.datasets import load_iris
        from sklearn import tree
        iris = load_iris()
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(iris.data, iris.target)
        # from sklearn.externals.six import StringIO
        with open("iris.dot", 'w') as f:
            f = tree.export_graphviz(clf, out_file=f)

        # brew install graphviz
        from os import system
        system("dot -Tpdf iris.dot -o iris.pdf")
        # from subprocess import call
        # call(["dot", "-Tpdf", "iris.dot", "-o", "iris.pdf"])

    def t2():
        import numpy as np
        from sklearn.tree import DecisionTreeClassifier
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])
        clf = DecisionTreeClassifier()
        clf.fit(X, y)
        importances = clf.feature_importances_
        print(importances)

    def t3():
        import pandas
        df = pandas.read_csv('train.csv', index_col='PassengerId')
        d = df[['Pclass', 'Fare', 'Age', 'Sex', 'Survived']].dropna().replace('male', 1).replace('female', 0)
        # print(d1)
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier()
        clf.random_state = 241
        clf.fit(d[['Pclass', 'Fare', 'Age', 'Sex']], d['Survived'])
        importances = clf.feature_importances_
        print(importances)
        # [ 0.13700004  0.31259037  0.24989737  0.30051221]
        # [ 0.14000522  0.30343647  0.2560461   0.30051221]
        pf('2', 'Fare Sex')

    t3()


def lesson3():
    def t1():
        X = [[0], [1], [2], [3]]
        y = [0, 0, 1, 1]
        from sklearn.neighbors import KNeighborsClassifier
        neigh = KNeighborsClassifier(n_neighbors=3)
        neigh.fit(X, y)
        KNeighborsClassifier(...)
        print(neigh.predict([[1.1]]))
        print(neigh.predict_proba([[0.9]]))

    def t2():
        import numpy as np
        from sklearn.cross_validation import KFold
        X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
        y = np.array([1, 2, 3, 4])
        kf = KFold(4, n_folds=2)
        print(kf)
        for train_index, test_index in kf:
            print("TRAIN:", train_index, "TEST:", test_index)
            # X_train, X_test = X[train_index], X[test_index]
            # y_train, y_test = y[train_index], y[test_index]

    def t3():
        def a(x, y):
            # 1.3
            from sklearn import cross_validation
            kf = cross_validation.KFold(len(y), n_folds=5, shuffle=True, random_state=42)
            # print(kf)

            # 1.4
            # from sklearn import svm
            # clf = svm.SVC(kernel='linear', C=1)
            from sklearn.neighbors import KNeighborsClassifier
            def test(i):
                clf = KNeighborsClassifier(n_neighbors=i)
                scores = cross_validation.cross_val_score(clf, x, y, cv=kf)
                score = sum(scores) / len(scores)
                # print(i, score, scores)
                return score

            l = list(map(test, range(1, 51)))
            m = max(l)
            return l.index(m) + 1, round(m, 2)

        # 1.1
        import pandas
        d = pandas.read_csv('wine.data.txt', header=None)

        # 1.2
        y = d[0]
        x = d.drop([0], axis=1)

        # 1.4
        i, m = a(x, y)
        pf('3_1', i)
        pf('3_2', m)

        # 1.5
        from sklearn.preprocessing import scale
        xs = scale(x)
        i, m = a(xs, y)
        pf('3_3', i)
        pf('3_4', m)
        # print(xs)

    t3()


def lesson4():
    def t1():
        # 1
        # import sklearn
        import numpy
        from sklearn.datasets import load_boston
        d = load_boston()
        xr = d.data
        y = d.target
        # print(y, x)

        # 2
        from sklearn.preprocessing import scale
        x = scale(xr)

        # 3
        from sklearn import cross_validation, neighbors
        kf = cross_validation.KFold(len(y), n_folds=5, shuffle=True, random_state=42)

        def test(p):
            clf = neighbors.KNeighborsRegressor(n_neighbors=5, weights='distance', metric='minkowski', p=p)
            scores = cross_validation.cross_val_score(clf, x, y, cv=kf, scoring='mean_squared_error')
            score = sum(scores) / len(scores)
            # print(p, score, scores)
            return score

        from numpy import linspace
        ap = linspace(1, 10, num=200)
        # print(ap)
        l = list(map(test, ap))
        m = max(l)
        pf('4', round(ap[l.index(m)], 1))
        print(m, l)
        # return l.index(m) + 1, round(m, 2)

    t1()


def lesson5():
    import numpy as np
    def t1():
        from sklearn.linear_model import Perceptron
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])
        clf = Perceptron()
        clf.fit(X, y)
        predictions = clf.predict(X)
        print(predictions)

    def t2():
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train = np.array([[100.0, 2.0], [50.0, 4.0], [70.0, 6.0]])
        X_test = np.array([[90.0, 1], [40.0, 3], [60.0, 4]])
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        print(X_train_scaled)
        print(X_test_scaled)

    def t():
        # 1
        from pandas import read_csv
        df = read_csv('perceptron-train.csv', header=None)
        dt = read_csv('perceptron-test.csv', header=None)
        yf = df[0]
        xf = df.drop([0], axis=1)
        # print(yf, xf)
        yt = dt[0]
        xt = dt.drop([0], axis=1)
        # print(yt, xt)

        # 2
        from sklearn.linear_model import Perceptron
        clf = Perceptron(random_state=241)
        clf.fit(xf, yf)
        af1 = clf.score(xf, yf)
        at1 = clf.score(xt, yt)
        rf = clf.predict(xf)
        rt = clf.predict(xt)
        # print(list(yf))
        # print(pf)
        # print(list(yt))
        # print(pt)

        # 3
        from sklearn.metrics import accuracy_score
        af = accuracy_score(yf, rf)
        at = accuracy_score(yt, rt)
        print(af, at)
        print(af1, at1)

        # 4
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        xfs = scaler.fit_transform(xf)
        xts = scaler.transform(xt)
        clf.fit(xfs, yf)
        afs1 = clf.score(xfs, yf)
        ats1 = clf.score(xts, yt)
        pfs = clf.predict(xfs)
        pts = clf.predict(xts)
        afs = accuracy_score(yf, pfs)
        ats = accuracy_score(yt, pts)
        print(afs, ats)
        print(afs1, ats1)
        pf('5', round(ats - at, 3))

    t()


def lesson6():
    def t1():
        import numpy as np
        X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
        y = np.array([1, 1, 2, 2])
        from sklearn.svm import SVC
        clf = SVC()
        clf.fit(X, y)
        print(clf)
        print(clf.predict([[-0.8, -1]]))
        print(clf.support_)

    def t():
        # 1
        from pandas import read_csv
        df = read_csv('svm-data.csv', header=None)
        y = df[0]
        x = df.drop([0], axis=1)
        # print(df)

        # 2
        from sklearn.svm import SVC
        clf = SVC(C=100000, random_state=241)
        clf.fit(x, y)
        print(clf.support_)
        pf('6', ' '.join([str(x + 1) for x in clf.support_]))

    t()


def lesson7():
    import pickle
    def pd(f, a):
        with open(f, 'wb') as handle:
            pickle.dump(a, handle)

    def pl(f):
        with open(f, 'rb') as handle:
            return pickle.load(handle)

    def t1():
        a = {'hello': 'world'}
        pd('70.p', a)
        print(a == pl('70.p'))

    def t():
        from sklearn import datasets, feature_extraction, svm, grid_search, cross_validation

        # 1
        newsgroups = datasets.fetch_20newsgroups(
            subset='all',
            categories=['alt.atheism', 'sci.space']
        )
        xf = newsgroups.data
        y = newsgroups.target
        # print(len(xf), yf, len(yf))

        # 2
        v = feature_extraction.text.TfidfVectorizer()
        x = v.fit_transform(xf)
        vn = v.get_feature_names()
        # print(vn)

        # 3
        import numpy as np
        grid = {'C': np.power(10.0, np.arange(-5, 6))}
        # print(grid)

        def sg():
            cv = cross_validation.KFold(y.size, n_folds=5, shuffle=True, random_state=241)
            clf = svm.SVC(kernel='linear', random_state=241)
            gs = grid_search.GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
            gs.fit(x, y)
            r = gs.grid_scores_
            pd('72.p', r)
            # print(r.all() == pl('71.p').all())
            print(r)

        # sg()
        r = pl('72.p')

        # print(r)
        # for a in r:
            # a.mean_validation_score — оценка качества по кросс-валидации
            # a.parameters — значения параметров
            # print(a.mean_validation_score, a.parameters)
            # print(str(a.mean_validation_score))
        # rm = list(map((lambda x: (x.mean_validation_score, x.parameters['C'])), r))
        ra = list(map((lambda x: x.mean_validation_score), r))
        rm = max(ra)
        ri = ra.index(rm)
        c = grid['C'][ri]
        # print(rm, c)
        # print(ra)

        # 4
        clf = svm.SVC(C=c, kernel='linear', random_state=241)
        clf.fit(x, y)
        co = clf.coef_

        # 5

        # top = np.argsort(co[0]) #[-10:]
        # top = abs(co.todense()).argpartition(-10)[-10:]
        top = np.argsort(np.absolute(np.asarray(co.todense())).reshape(-1))[-10:]
        # print(co[0])
        # print(top)
        names = [vn[x] for x in top]
        print(names)
        print(names.sort())
        out = ' '.join(names)
        # print(out)
        # cc = clf.support_
        pf('7', out)

    t()

def lesson8():
    def t1():
        import numpy as np
        from sklearn.metrics import roc_auc_score
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.4, 0.35, 0.8])
        r = roc_auc_score(y_true, y_scores)
        print(r)

    def t():
        # 1
        from pandas import read_csv
        # from numpy import linalg, sqrt
        from math import exp, sqrt
        df = read_csv('data-logistic.csv', header=None)
        y = df[0]
        x = df.drop([0], axis=1)
        # print(df)

        # 2-3-4
        L = len(y)
        S = 1e-5
        def g(steps = 10000, k = 0.1, C = 0):
            (w1, w2) = (0, 0)
            s = 0
            while s < steps:
                i = 0
                (d1, d2) = (0, 0)
                while i < L:
                    d = 1 - 1 / (1 + exp(-y[i] * (w1 * x[1][i] + w2 * x[2][i])))
                    d1 = d1 + y[i] * x[1][i] * d
                    d2 = d2 + y[i] * x[2][i] * d
                    i = i + 1
                _w1 = w1 + (k / L) * d1 - k * C * w1
                _w2 = w2 + (k / L) * d2 - k * C * w2
                e = sqrt((_w1 - w1)**2 + (_w2 - w2)**2)
                (w1, w2) = (_w1, _w2)
                # print(s, e, w1, w2)
                if e < S:
                    break
                s = s + 1
            print(k, C, s, e, w1, w2)
            return w1, w2

        # 5
        from sklearn.metrics import roc_auc_score as auc
        a = lambda x1, x2: 1 / (1 + exp(-w1 * x1 - w2 * x2))
        al = lambda: list(map(a, x[1], x[2]))
        au = lambda: str(round(auc(y, al()), 3))
        (w1, w2) = g()
        a1 = au()
        (w1, w2) = g(C=10)
        a2 = au()
        pf('8', a1 + ' ' + a2)

        # # ||w||
        # # norm = numpy.linalg.norm( (w1, w2) )
        # norm = numpy.linalg.norm([w1, w2])
        # e2 = linalg.norm([[_w1, _w2], [w1, w2]])


    def t2():
        print()

    t()

start_time = time.time()
lesson8()
print("--- %s seconds ---" % str(time.time() - start_time))
# print(timeit.timeit(lesson7(), number=10000)
