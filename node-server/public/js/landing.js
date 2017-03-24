angular.module("nautilusLanding",[])

  .controller("landingController", ['$scope','$http','$window', function($scope,$http,$window){

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

  $scope.callGetColumns = function() {
    $scope.currentModelName = $scope.mod; // use this var in trained.js to retrieve the particular model using getColumns API call
    $window.location.href = '/trained.html';
  }

}])
