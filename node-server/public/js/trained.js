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
		var data = {
			dataJson: $scope.pval,
			modelName: $scope.$parent.modelName,
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
			$scope.result = parseInt(res.data)
			if($scope.result == 0) $scope.recommended = true;
			else $scope.recommended = false;
			$scope.popup = true;

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
