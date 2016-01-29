p = '/Users/malex/Dropbox/create/project/ML/l1/'


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


def t2():
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])
    clf = DecisionTreeClassifier()
    clf.fit(X, y)
    importances = clf.feature_importances_
    print(importances)

def pf(name, value):
    f = open(name + '.txt', 'w')
    f.write(str(value))
    f.close()
    print(name + '="%s"\n' % value)


def main():
    import pandas
    data = pandas.read_csv(p + 'train.csv', index_col='PassengerId')

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


# main()
t2()
