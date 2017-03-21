angular.module("nautilus-learn", ['ngRoute','ngFileUpload'])
  .config(function($routeProvider){
    $routeProvider
			.when('/step-1', {
				templateUrl : '../templates/step-1.html',
				controller : 'UploadController'
			})
			.when('/step-2', {
				templateUrl : '../templates/step-2.html',
				controller : 'PredictionChoiceController'
			})
			.when('/step-3', {
				templateUrl : '../templates/step-3.html',
				controller : 'FeatureSelectionController'
			})
			.when('/step-4', {
				templateUrl : '../templates/step-4.html',
				controller : 'FinishController'
			})
			.otherwise('/step-1')
  })

  .controller("StepController", ['$scope', '$location', function($scope, $location){
	  $scope.currentStep = 1;

    $scope.startOver = function(){
      $location.url('step-1');
    }

  }])
  .controller("UploadController", ['$scope','Upload', '$timeout', '$location', function($scope, Upload, $timeout, $location){
      $scope.fileNotUploaded = true;
      $scope.toStepTwo = function(){
          $location.url('step-2');
      }
      $scope.$parent.currentStep = 1;

	  $scope.uploadFiles = function(file, errFiles) {
        $scope.f = file;
        console.log(file);
        $scope.errFile = errFiles && errFiles[0];
        if (file) {
            file.upload = Upload.upload({
                url: '/upload',
                data: {file: file}
            });

            file.upload.then(function (response) {
                $timeout(function () {
                    file.result = response.data;
                });
            }, function (response) {
                if (response.status > 0)
                    $scope.errorMsg = response.status + ': ' + response.data;
            }, function (evt) {
                file.progress = Math.min(100, parseInt(100.0 * evt.loaded / evt.total));
                if(file.progress == 100)
                    $scope.fileNotUploaded = false;
            });
        }
    }

  }])

  .controller("PredictionChoiceController", ['$scope', '$http', '$location', 'memoryFactory', function($scope, $http, $location, $memoryFactory ){
      $scope.$parent.currentStep = 2;

      $scope.predFeature = "";
      $scope.predType = $memoryFactory.getPredType();

      $scope.features = {};
      $scope.filepath = "";
      $http.get("/get-file-path", {}).then(function(response){
          console.log(response.data.filepath)
          $scope.filepath = response.data.filepath;
          $http.get("http://127.0.0.1:8000/Engine/getFeatures/?path=" + response.data.filepath, {}).then(function(response){
                console.log(response)
                $scope.features = response.data.keys;
                $scope.predFeature = response.data.keys[0];
            }, function(response){
                console.log("PredictionChoiceController: ", response);
            })
      }, function(response){
          console.log(response)
      });

      $scope.toStepThree = function(){
          // console.log("dfjadkfjkadjfkjad")
          $memoryFactory.setPredFeature($scope.predFeature);
          $memoryFactory.setPredType($scope.predType);
          console.log("PredType : " + $memoryFactory.getPredType())
          $location.url('step-3');
      }

  }])

  .controller("FeatureSelectionController", ['$scope', '$http', '$location', 'memoryFactory', function($scope, $http, $location, $memoryFactory){
      $scope.$parent.currentStep = 3;

      $scope.features = {};
      $scope.filepath = "";
      $http.get("/get-file-path", {}).then(function(response){
          console.log(response.data.filepath)
          $scope.filepath = response.data.filepath;
          var vals;
          if ($memoryFactory.getPredType() == 0) vals = "pvals"; else vals = "fvals";
          console.log("Vals value: *****: ", vals);
          $http.get("http://127.0.0.1:8000/Engine/" + vals + "/?path=" + response.data.filepath + "&feature=" + $memoryFactory.getPredFeature(), {}).then(function(response){
                console.log("Feature Selection Controller: ", response);
                $scope.features = response.data;
            }, function(response){
                console.log(response);
            })
      }, function(response){
          console.log(response)
      });

      $scope.toStepFour = function(){
          var arr = [];
          for(var i = 0; i < $scope.features.length; i++){
              if($scope.features[i].selected === true) arr.push($scope.features[i].key);
          }

          console.log(arr)
          console.log("Scope modelName: ", $scope.modelName);
        //  console.log($scope.$parent.modelName);
          var data = {
              path : $scope.filepath,
              keys : arr,
              feature: $memoryFactory.getPredFeature(),
              modelName: $scope.modelName,
              userName: ""
          }
          var type = "";
          if ($memoryFactory.getPredType() == 0) type = "Class"; else type = "Regression";



        console.log("Initiating...");
        var apiUrl = "http://127.0.0.1:8000/Engine/buildModel" + type + "/?data=";

        /* $http.get(apiurl + JSON.stringify(data),{}).then(function(res){
             $memoryFactory.setResult(res.data.result)
             console.log("Result : "  + res.data.result)
             $location.url('step-4');
         },function(res){
             console.log('err : ' + res)
         }) */

         var fullUrl = apiUrl + JSON.stringify(data);
         var tempData = {
           urlData: data,
           urlType: type,
         };
         console.log("Making node call");

  $http({
    method: 'POST',
    url: 'http://localhost:3000/call-build-model',
    data: 'data='+JSON.stringify(tempData),
    headers: {'Content-Type': 'application/x-www-form-urlencoded'}
  }).then(function(res){
    console.log("call-build-model response: ", res);
    response = JSON.parse(res.data);
    console.log("Parsed response: ", response);
    responseBody = JSON.parse(response.body);
    console.log("Parsed response body: ", responseBody);

    $memoryFactory.setResult(responseBody.result)
    console.log("Result : "  + responseBody.result)
    $location.url('step-4');
  },function(res){
      console.log('err : ' + res)
  })

        /* $http.post("/call-build-model", tempData, {}).then(function(res){
           $memoryFactory.setResult(res.data.result)
           console.log("Result : "  + res.data.result)
           $location.url('step-4');
         }) */

         console.log("Finished node call");

         $http.post("/save-selected-feat", data, {}).then(function(res){
           console.log(res)
         },function(res){
             console.log('err : ' + res)
         })
      }

      $scope.toggleClass = function(feat){
          feat.selected = !feat.selected;
      }


  }])


.controller("FinishController", ['$scope', 'memoryFactory','$window', function($scope, $memoryFactory, $window){
    $scope.$parent.currentStep = 4;
    console.log($memoryFactory.getResult())
    $scope.result  = $memoryFactory.getResult();
    $scope.predType = $memoryFactory.getPredType();
    $scope.dispVal = "";

    if($scope.predType == 0) $scope.dispVal = "Accuracy"; else $scope.dispVal = "Mean Squared Error";

    $scope.finish = function(){
      $window.location.href = '/trained.html';
    }
}])

.factory('memoryFactory', function() {
  var fac={}
  var accuracy = 0;
  var predFeature = "";
  var predType = 0;

  fac.setResult = function(acc){
    accuracy = acc;
  }

  fac.getResult = function(){
    return accuracy;
  }

  fac.setPredFeature = function(feat){
    predFeature = feat;
  }

  fac.getPredFeature = function(){
    return predFeature;
  }

  fac.setPredType = function(pred){
    predType = pred;
  }

  fac.getPredType = function(){
    return predType;
  }

  return fac;
});
