from lib import *

def main():
    from pandas import read_csv
    from sklearn import feature_extraction
    from scipy.sparse import hstack
    from sklearn.linear_model import Ridge
    # 1
    x = read_csv('salary-train.csv')
    cf, cl, cc, cs = 'FullDescription', 'LocationNormalized', 'ContractTime', 'SalaryNormalized'
    # print(x.head())

    # 2.1/2
    xf = x[cf].str.lower().replace('[^a-zA-Z0-9]', ' ', regex=True)

    # 2.3
    v = feature_extraction.text.TfidfVectorizer(min_df=5)
    xv = v.fit_transform(xf)

    # 2.4
    x[cl].fillna('nan', inplace=True)
    x[cc].fillna('nan', inplace=True)

    # 2.5
    d = feature_extraction.DictVectorizer()
    xd = d.fit_transform(x[[cl, cc]].to_dict('records'))

    # 2.6 Объедините все полученные признаки в одну матрицу "объекты-признаки".
    xm = hstack([xv, xd])

    # 3 Обучите гребневую регрессию с параметром alpha=1. Целевая переменная записана в столбце SalaryNormalized.
    clf = Ridge(alpha=1.0)
    clf.fit(xm, x[cs])

    # 4. Постройте прогнозы для двух примеров из файла salary-test-mini.csv.
    t = read_csv('salary-test-mini.csv')
    tf = t[cf].str.lower().replace('[^a-zA-Z0-9]', ' ', regex=True)
    tv = v.transform(tf)
    td = d.transform(t[[cl, cc]].to_dict('records'))
    p = clf.predict(hstack([tv, td]))
    pf('10', pp(p.round(2)))

start = time()
main()
finish(start)
# 45.09142208099365 seconds