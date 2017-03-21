from django.http import HttpResponse
import pandas as pd
import json
import pickle
import os
import threading
from sklearn.model_selection import train_test_split
from . import utils

def dirExists(request):
    userName = request.GET.get('userName')
    response = utils.createDir(userName)
    if response == True:
        return HttpResponse(json.dumps({'exists':True}))
    else:
        return HttpResponse(json.dumps({'exists':False}))

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

    lsvcThread = utils.ModelThread(1, XTrain, yTrain, XTest, yTest)
    dtThread = utils.ModelThread(2, XTrain, yTrain, XTest, yTest)
    rfThread = utils.ModelThread(3, XTrain, yTrain, XTest, yTest)
    adaThread = utils.ModelThread(4, XTrain, yTrain, XTest, yTest)
    lsvcThread.start()
    dtThread.start()
    rfThread.start()
    adaThread.start()
    # lsvc = LinearSVC()
    # lsvc.fit(XTrain, yTrain)
    # lsvcScore = lsvc.score(XTest, yTest)

    lsvcThread.join()
    dtThread.join()
    rfThread.join()
    adaThread.join()
    lsvcScore, lsvc = lsvcThread.getAnswer()
    if resAcc < lsvcScore:
        resAcc = lsvcScore
        model = lsvc

    # dt = DecisionTreeClassifier()
    # dt.fit(XTrain, yTrain)
    # dtScore = dt.score(XTest, yTest)
    dtScore, dt = dtThread.getAnswer()
    if resAcc < dtScore:
        resAcc = dtScore
        model = dt


    # rf = RandomForestClassifier()
    # rf.fit(XTrain, yTrain)
    # rfScore = rf.score(XTest, yTest)
    rfScore, rf = rfThread.getAnswer()
    if resAcc < rfScore:
        resAcc = rfScore
        model = rf

    # ada = AdaBoostClassifier()
    # ada.fit(XTrain, yTrain)
    # adaScore = ada.score(XTest, yTest)
    adaScore, ada = adaThread.getAnswer()
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
    lrThread = utils.ModelThread(5, XTrain, yTrain, XTest, yTest)
    rfThread = utils.ModelThread(6, XTrain, yTrain, XTest, yTest)
    gbThread = utils.ModelThread(7, XTrain, yTrain, XTest, yTest)

    lrThread.start()
    rfThread.start()
    gbThread.start()

    lrThread.join()
    rfThread.join()
    gbThread.join()
    # lr = LinearRegression()
    # lr.fit(XTrain, yTrain)
    # lrScore = mean_squared_error(lr.predict(XTest), yTest)
    lrScore, lr = lrThread.getAnswer()
    if resAcc > lrScore:
        resAcc = lrScore
        model = lrScore

    # rf = RandomForestRegressor()
    # rf.fit(XTrain, yTrain)
    # rfScore = mean_squared_error(rf.predict(XTest), yTest)
    rfScore, rf = rfThread.getAnswer()
    if resAcc > rfScore:
        resAcc = rfScore
        model = rf

    # gb = GradientBoostingRegressor()
    # gb.fit(XTrain, yTrain)
    # gbScore = mean_squared_error(gb.predict(XTest), yTest)
    gbScore, gb = gbThread.getAnswer()
    if resAcc > gbScore:
        resAcc = gbScore
        model = gb

    # print model
    # print resAcc
    resAccJson = {'result': resAcc}
    print("Result accuracy regression: ", resAccJson);
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

def getModels(request):
    userName = request.GET.get('userName')
    data = {'userName':userName}
    path = utils.findUserName(data)
    models = []
    for root, dirs, files in os.walk(path):
        for file_ in files:
            if '_cols' not in file_:
                models.append(file_.split('.')[0])
    # print(models)
    return HttpResponse(json.dumps({'result':models}))

def runModel(request):
    data = request.GET.get('data')
    data = json.loads(data)
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
