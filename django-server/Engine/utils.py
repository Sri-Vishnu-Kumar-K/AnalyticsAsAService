import os
import threading
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


class ModelThread(threading.Thread):
    def __init__(self, modelType, XTrain, yTrain, XTest, yTest):
        threading.Thread.__init__(self)
        self.modelType = modelType
        self.model = None
        self.score = None
        self.XTrain = XTrain
        self.yTrain = yTrain
        self.XTest = XTest
        self.yTest = yTest

    def run(self):
        if self.modelType == 1:
            lsvc = LinearSVC()
            lsvc.fit(self.XTrain, self.yTrain)
            lsvcScore = lsvc.score(self.XTest, self.yTest)
            self.score, self.model = lsvcScore, lsvc
        elif self.modelType == 2:
            dt = DecisionTreeClassifier()
            dt.fit(self.XTrain, self.yTrain)
            dtScore = dt.score(self.XTest, self.yTest)
            self.score, self.model = dtScore, dt
        elif self.modelType == 3:
            rf = RandomForestClassifier()
            rf.fit(self.XTrain, self.yTrain)
            rfScore = rf.score(self.XTest, self.yTest)
            self.score, self.model = rfScore, rf
        elif self.modelType == 4:
            ada = AdaBoostClassifier()
            ada.fit(self.XTrain, self.yTrain)
            adaScore = ada.score(self.XTest, self.yTest)
            self.score, self.model = adaScore, ada
        elif self.modelType == 5:
            lr = LinearRegression()
            lr.fit(self.XTrain, self.yTrain)
            lrScore = mean_squared_error(lr.predict(self.XTest), self.yTest)
            self.score, self.model = lrScore, lr
        elif self.modelType == 6:
            rf = RandomForestRegressor()
            rf.fit(self.XTrain, self.yTrain)
            rfScore = mean_squared_error(rf.predict(self.XTest), self.yTest)
            self.score, self.model = rfScore, rf
        elif self.modelType == 7:
            gb = GradientBoostingRegressor()
            gb.fit(self.XTrain, self.yTrain)
            gbScore = mean_squared_error(gb.predict(self.XTest), self.yTest)
            self.score, self.model = gbScore, gb

    def getAnswer(self):
        return self.score, self.model


def findUserName(data):

    userName = data['userName']
    # print(os.getcwd())
    cwdPath = os.getcwd()+'/UserData/'
    # print(cwdPath)
    # print(os.path.exists(cwdPath))
    path = ''
    for root, dirs, files in os.walk(cwdPath):
        for directory in dirs:
            # print(directory)
            if directory == userName:
                path = os.path.join(root,directory)
    return path

def createDir(name):
    root = os.getcwd()+'/UserData/'
    # print(os.path.exists(root))
    if os.path.exists(os.path.join(root,name)):
        return True
    os.mkdir(os.path.join(root,name))
    if os.path.exists(os.path.join(root,name)):
        return True
    else:
        return False
