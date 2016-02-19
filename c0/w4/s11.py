from lib import *

def main():
    # 1 Загрузите данные close_prices.csv. В этом файле приведены цены акций 30 компаний
    from pandas import read_csv
    x = read_csv('close_prices.csv')

    # 2 На загруженных данных обучите преобразование PCA с числом компоненты равным 10.
    #  Скольких компонент хватит, чтобы объяснить 90% дисперсии?
    xd = x.drop(['date'], axis=1)
    from sklearn.decomposition import PCA
    p1 = PCA(n_components=0.9)
    p1.fit(xd)
    # print(pca.explained_variance_ratio_)
    n = p1.n_components_
    pf('11_1', n)

    # 3 Примените построенное преобразование к исходным данным и возьмите значения первой компоненты.
    p2 = PCA(n_components=n)
    p2.fit(xd)
    xt = p2.transform(xd)
    x1 = xt[:, 0]
    # print(xt[:, 0])

    # 4 Загрузите информацию об индексе Доу-Джонса из файла djia_index.csv.
    #  Чему равна корреляция Пирсона между первой компонентой и индексом Доу-Джонса?
    y = read_csv('djia_index.csv')
    y1 = y['^DJI']
    print(x1.shape, y1.shape)
    from numpy import corrcoef
    c = corrcoef(x1, y1)
    pf('11_2', round(c[0][1], 2))

    # 5
    m = zip(p2.components_[0], xd.columns)
    pf('11_3', max(m)[1])

start = time()
main()
finish(start)
