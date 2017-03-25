angular.module("nautilusTrain",['ngRoute'])
.config(function($routeProvider){
	$routeProvider
		.when('/landing-1', {
			templateUrl : '../templates/landing-1.html',
			controller : 'landingController'
		})
		.when('/landing-2', {
			templateUrl : '../templates/landing-2.html',
			controller : 'ScoreCheckerController'
		}).otherwise('/landing-1')
	})

		.controller("landingMainController", ['$scope','$location', '$http', '$window', function($scope, $location,$http, $window){
			console.log('init /land-1');
			$location.url('landing-1');
		}])
		.controller("landingController", ['$scope','$location','memoryFactory','$http','$window', function($scope,$location,memoryFactory,$http,$window){

var tempData = {
	userName: ""
};

$http({
	method: 'POST',
	url: 'http://localhost:3000/call-get-models',
	data: 'data='+JSON.stringify(tempData),
	headers: {'Content-Type': 'application/x-www-form-urlencoded'}
}).then(function(res){
	console.log("call-get-models response: ", res);
	 response = JSON.parse(res.data);
	 responseBody = JSON.parse(response.body);
	 console.log("Parsed response body: ", responseBody);
	 var modelsArrJson = responseBody.result;
	 var userName = responseBody.userName;

	 $scope.models = modelsArrJson;
	/* console.log("Parsed response: ", response);
	responseBody = JSON.parse(response.body);
	console.log("Parsed response body: ", responseBody); */

/*  $memoryFactory.setResult(responseBody.result)
	console.log("Result : "  + responseBody.result)
	$location.url('step-4'); */
},function(res){
		console.log('err : ' + res)
})

$scope.callGetColumns = function(name) {
	console.log(name);
	memoryFactory.setModelName(name); // use this var in trained.js to retrieve the particular model using getColumns API call
	$location.url('/landing-2');
}

}])
.controller("ScoreCheckerController", ['$scope','memoryFactory', '$http', '$window', function($scope, memoryFactory ,$http, $window){

// $scope.features = []
$scope.pval = {}

$scope.popup = false;
$scope.recommended = true;

/*$http.get("/get-selected-feat", {}).then(function(res){
// $scope.features = res.data.keys
for(var i=0 ; i< res.data.keys.length; i++)
	$scope.pval[res.data.keys[i]] = "";
console.log(res.data.keys)
}, function(res){
console.log("err : " + res)
})*/
var getColsData = {
userName: "",
modelName: memoryFactory.getModelName()//$scope.$parent.modelName,
};

console.log("getColsData: ", getColsData);
$http({
method: 'POST',
url: 'http://localhost:3000/call-get-columns',
data: 'data='+JSON.stringify(getColsData),
headers: {'Content-Type': 'application/x-www-form-urlencoded'}
}).then(function(res){

console.log("call-get-columns response: ", res);
resData = JSON.parse(res.data);
console.log("resData: ", resData);
resBody = JSON.parse(resData.body);
console.log("resBody: ", resBody);
var parsingArray = resBody.result;
console.log("parsingArray: ", parsingArray);
for(var i=0 ; i< parsingArray.length; i++)
	$scope.pval[parsingArray[i]] = "";
console.log(parsingArray);
},function(res){
	console.log('err : ' + res);
})
$scope.result = 0;

$scope.scoreChecker = function(){
var data = {
	dataJson: $scope.pval,
	modelName: memoryFactory.getModelName(),
	userName: ""
}
$http({
	method: 'POST',
	url: 'http://localhost:3000/call-run-model',
	data: 'data='+JSON.stringify(data),
	headers: {'Content-Type': 'application/x-www-form-urlencoded'}
}).then(function(res){
	/* console.log("call-build-model response: ", res);
	response = JSON.parse(res.data);
	console.log("Parsed response: ", response);
	responseBody = JSON.parse(response.body);
	console.log("Parsed response body: ", responseBody);

	$memoryFactory.setResult(responseBody.result)
	console.log("Result : "  + responseBody.result)
	$location.url('step-4'); */
	resData = JSON.parse(res.data);
	console.log(resData.body);
	$scope.result = parseInt(resData.body);
	data.userName = "yourName";
	$scope.apiLink = "http://localhost:8000/Engine/runModel/?data="+JSON.stringify(data);
	$window.alert(parseInt(resData.body));
	/*if($scope.result == 0) $scope.recommended = true;
	else $scope.recommended = false;
	$scope.popup = true;*/

},function(res){
		console.log('err : ' + res)
})

/* $http.get("http://127.0.0.1:8000/Engine/runModel/?data=" + JSON.stringify($scope.pval), {}).then(function(res){
	$scope.result = parseInt(res.data)
	if($scope.result == 0) $scope.recommended = true;
	else $scope.recommended = false;
	$scope.popup = true;

}, function(res){

}) */

}

$scope.dismissPopup = function(){
$scope.popup = false;
}

$scope.startOver = function(){
$window.location.href = '/learn.html';
}

}])
.factory('memoryFactory',function(){
	var data = {};
	var modelName = '';

	data.setModelName = function(name){
		modelName = name;
	}
	data.getModelName = function(){
		return modelName;
	}
	return data;
  }
)
