from django.http import HttpResponse
import pandas as pd
import json
import pickle
import os

from . import utils

def getFeatures(request):
    dataPath = request.GET.get('path')
    df = pd.read_csv(dataPath)
    cols = []
    for col in df.columns:
        cols.append(col)
    retval = {"keys": cols}
    return HttpResponse(json.dumps(retval))

def fvals(request):
    from sklearn.feature_selection import mutual_info_regression
    dataPath = request.GET.get('path')
    featureName = request.GET.get('feature')
    # print path
    # print feature_name
    df = pd.read_csv(dataPath)
    df.set_index('ID', inplace=True)
    df.fillna(value=0, inplace=True)
    cols = []
    for col in df.columns:
        cols.append(col)
    cols.remove(featureName)
    mival = mutual_info_regression(df[cols], df[featureName])
    maxMival = max(mival)
    mivalDict = [{'key': cols[i], 'pvalue': (mival[i]/maxMival) * 100.0, 'selected': False} for i in
                range(0, len(cols))]
    # print json.dumps(PvalDict)
    return HttpResponse(json.dumps(mivalDict))


def pvals(request):
    from sklearn.feature_selection import chi2
    dataPath = request.GET.get('path')
    featureName = request.GET.get('feature')
    # print path
    # print feature_name
    df = pd.read_csv(dataPath)
    df.set_index('ID', inplace=True)
    df.fillna(value=0, inplace=True)
    cols = []
    for col in df.columns:
        cols.append(col)
    cols.remove(featureName)
    for col in cols:
        if min(df[col]) < 0:
            adder = -1 * min(df[col])
        else:
            adder = min(df[col])
        df.loc[:, col] += adder
    chi2val, pval = chi2(df[cols], df[featureName])

    PvalDict = [{'key': cols[i], 'pvalue': (1.0 - pval[i]) * 100.0, 'selected': False} for i in
                range(0, len(cols))]
    # print json.dumps(PvalDict)
    return HttpResponse(json.dumps(PvalDict))


def buildModelClass(request):
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.svm import LinearSVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    retVal = request.GET.get('data')

    data = json.loads(retVal)

    dataPath = data['path']
    featureName = data['feature']
    selectVars = data['keys']
    modelName = data['modelName']

    path = utils.findUserName(data)
    if path=='':
        #print('Path not found Error')
        return HttpResponse(json.dumps({'result':'Username not found!'}))

    if os.path.exists(os.path.join(path,modelName)+'.pickle'):
        return HttpResponse(json.dumps({'result':'Model Name exists'}))

    f = open(os.path.join(path,modelName)+'_cols.pickle', 'wb')
    pickle.dump(selectVars, f)
    f.close()

    df = pd.read_csv(dataPath)
    df.set_index('ID', inplace=True)

    X = df[selectVars]
    y = df[featureName]

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

    # print model
    # print resAcc
    resAccJson = {'result': resAcc}
    # print(resAcc)
    # print(path+modelName)
    f = open(os.path.join(path,modelName)+'.pickle', 'wb')
    pickle.dump(model, f)
    f.close()
    return HttpResponse(json.dumps(resAccJson))


def buildModelRegression(request):
    from sklearn.metrics import mean_squared_error
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    retVal = request.GET.get('data')

    data = json.loads(retVal)
    dataPath = data['path']
    featureName = data['feature']
    selectVars = data['keys']
    modelName = data['modelName']

    path = utils.findUserName(data)

    if path=='':
        #print('Path not found Error')
        return HttpResponse(json.dumps({'result':'Username not found!'}))

    if os.path.exists(os.path.join(path,modelName)+'.pickle'):
        return HttpResponse(json.dumps({'result':'Model Name exists'}))


    f = open(os.path.join(path,modelName)+'_cols.pickle', 'wb')
    pickle.dump(selectVars, f)
    f.close()

    df = pd.read_csv(dataPath)
    df.set_index('ID', inplace=True)

    X = df[selectVars]
    y = df[featureName]

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

    # print model
    # print resAcc
    resAccJson = {'result': resAcc}
    f = open(os.path.join(path,modelName)+'.pickle', 'wb')
    pickle.dump(model, f)
    f.close()
    return HttpResponse(json.dumps(resAccJson))

def getColumns(request):
    data = request.GET.get('data')

    data = json.loads(data)
    modelName = data['modelName']
    path = utils.findUserName(data)

    if path=='':
        #print('Path not found Error')
        return HttpResponse(json.dumps({'result':'Username not found!'}))

    if os.path.exists(os.path.join(path,modelName)+'.pickle') == False:
        return HttpResponse(json.dumps({'result':'Model does not exist'}))

    # print(os.path.join(path,modelName))
    f = open(os.path.join(path,modelName)+'_cols.pickle','rb')
    selectVars = pickle.load(f)
    f.close()
    return HttpResponse(json.dumps({'result':selectVars}))


def runModel(request):
    data = request.GET.get('data')
    dataJson = data['dataJson']
    modelName = data['modelName']
    path = utils.findUserName(data)
    if path=='':
        #print('Path not found Error')
        return HttpResponse(json.dumps({'result':'Username not found!'}))

    if os.path.exists(os.path.join(path,modelName)+'.pickle') == False:
        return HttpResponse(json.dumps({'result':'Model does not exist'}))

    # print(os.path.join(path,modelName))
    f = open(os.path.join(path,modelName)+'.pickle','rb')
    model = pickle.load(f)
    f.close()
    # print data
    df = pd.read_json(dataJson, orient='index', typ='Series')
    f = open(os.path.join(path,modelName)+'_cols.pickle','rb')
    selectVars = pickle.load(f)
    f.close()

    res = model.predict(df[selectVars].reshape(1, -1))
    return HttpResponse(str(res[0]))
