from django.http import HttpResponse
import pandas as pd
import json
import pickle


def getFeatures(request):
    path = request.GET.get('path')
    print path
    df = pd.read_csv(path)
    cols = []
    for col in df.columns:
        cols.append(col)
    retval = {"keys":cols}
    return HttpResponse(json.dumps(retval))


def pvals(request):
    from sklearn.feature_selection import chi2
    path = request.GET.get('path')
    feature_name = request.GET.get('feature')
    print path
    print feature_name
    df = pd.read_csv(path)
    df.set_index('ID', inplace=True)
    df.fillna(value=0, inplace=True)
    cols = []
    for col in df.columns:
        cols.append(col)
    cols.remove(feature_name)
    for col in cols:
        if min(df[col]) < 0:
            adder = -1 * min(df[col])
        else:
            adder = min(df[col])
        df.loc[:, col] += adder
    chi2val, pval = chi2(df[cols], df[feature_name])

    PvalDict = [{'key': cols[i], 'pvalue': (1.0 - pval[i]) * 100.0, 'selected': False} for i in
                range(0, len(cols))]
    print json.dumps(PvalDict)
    return HttpResponse(json.dumps(PvalDict))


def buildModelClass(request):
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.cross_validation import train_test_split
    from sklearn.svm import LinearSVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    retVal = request.GET.get('data')
    data = json.loads(retVal)
    path = data['path']
    feature_name = data['feature']
    selectVars = data['keys']
    df = pd.read_csv(path)
    df.set_index('ID', inplace=True)

    X = df[selectVars]
    y = df[feature_name]

    f = open('cols.pickle', 'w')
    pickle.dump(selectVars, f)
    f.close()

    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.3)

    resAcc = 0
    model = None

    lsvc = LinearSVC()
    lsvc.fit(XTrain, yTrain)
    lsvcScore = lsvc.score(XTest, yTest)
    if resAcc < lsvcScore:
        resAcc = lsvcScore
        model = lsvc

    dt = DecisionTreeClassifier()
    dt.fit(XTrain, yTrain)
    dtScore = dt.score(XTest, yTest)
    if resAcc < dtScore:
        resAcc = dtScore
        model = dt
    rf = RandomForestClassifier()
    rf.fit(XTrain, yTrain)
    rfScore = rf.score(XTest, yTest)
    if resAcc < rfScore:
        resAcc = rfScore
        model = rf

    ada = AdaBoostClassifier()
    ada.fit(XTrain, yTrain)
    adaScore = ada.score(XTest, yTest)
    if resAcc < adaScore:
        resAcc = adaScore
        model = ada

    print model
    print resAcc
    resAccJson = {'result': resAcc}
    f = open('FinalPickle.pickle', 'w')
    pickle.dump(model, f)
    f.close()
    return HttpResponse(json.dumps(resAccJson))


def buildModelRegression(request):
    from sklearn.metrics import mean_squared_error
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
    from sklearn.cross_validation import train_test_split
    retVal = request.GET.get('data')
    data = json.loads(retVal)
    path = data['path']
    selectVars = data['keys']
    feature_name = data['feature']
    df = pd.read_csv(path)
    df.set_index('ID', inplace=True)

    X = df[selectVars]
    y = df[feature_name]

    f = open('cols.pickle', 'w')
    pickle.dump(selectVars, f)
    f.close()

    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.3)

    resAcc = 0
    model = None

    lr = LinearRegression()
    lr.fit(XTrain, yTrain)
    lrScore = mean_squared_error(lr.predict(XTest), yTest)
    if resAcc > lrScore:
        resAcc = lrScore
        model = lrScore

    rf = RandomForestRegressor()
    rf.fit(XTrain, yTrain)
    rfScore = mean_squared_error(rf.predict(XTest), yTest)
    if resAcc > rfScore:
        resAcc = rfScore
        model = rf

    gb = GradientBoostingRegressor()
    gb.fit(XTrain, yTrain)
    gbScore = mean_squared_error(gb.predict(XTest), yTest)
    if resAcc > gbScore:
        resAcc = gbScore
        model = gb

    print model
    print resAcc
    resAccJson = {'result': resAcc}
    f = open('FinalPickle.pickle', 'w')
    pickle.dump(model, f)
    f.close()
    return HttpResponse(json.dumps(resAccJson))


def runModel(request):
    data = request.GET.get('json')
    print data
    df = pd.read_json(data, orient='index', typ='Series')
    f = open('FinalPickle.pickle', 'r')
    model = pickle.load(f)
    f.close()
    f = open('cols.pickle', 'r')
    Vars = pickle.load(f)
    f.close()
    res = model.predict(df[Vars].reshape(1, -1))
    return HttpResponse(str(res[0]))
