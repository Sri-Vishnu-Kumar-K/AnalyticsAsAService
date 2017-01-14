angular.module("nautilusTrain",[])

.controller("ScoreCheckerController", ['$scope', '$http', '$window', function($scope, $http, $window){
	
	// $scope.features = []
	$scope.pval = {}
	
	$scope.popup = false;
	$scope.recommended = true;
	
	$http.get("/get-selected-feat", {}).then(function(res){
		// $scope.features = res.data.keys
		for(var i=0 ; i< res.data.keys.length; i++) 
			$scope.pval[res.data.keys[i]] = "";
		console.log(res.data.keys)
	}, function(res){
		console.log("err : " + res)
	})
	
	$scope.result = 0;
	
	$scope.scoreChecker = function(){
		$http.get("http://127.0.0.1:8000/Engine/runModel/?json=" + JSON.stringify($scope.pval), {}).then(function(res){
			$scope.result = parseInt(res.data)
			if($scope.result == 0) $scope.recommended = true;
			else $scope.recommended = false;
			$scope.popup = true;
			
		}, function(res){
			
		})
		
	}
	
	$scope.dismissPopup = function(){
		$scope.popup = false;
	}
	
	$scope.startOver = function(){
		$window.location.href = '/learn.html';
	}
	
}])