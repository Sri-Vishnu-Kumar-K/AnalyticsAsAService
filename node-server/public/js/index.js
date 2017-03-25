angular.module("nautilus",[])

  .controller("LoginController", ['$scope', '$http', '$window', function($scope, $http, $window){

    $scope.data = {
      username: "",
      password: ""
    }

    $scope.loginFailed = false;

    var config = {}

    $scope.loginCall= function(){
      console.log($scope.data);
      $http.post('/login', $scope.data, config).then($scope.successCallback, $scope.errorCallback);
    }

    $scope.successCallback = function(response){
      console.log(response.data)
      if(response.data.status === true) {
        if(response.data.pristine === true)
          $window.location.href = '/trained.html';
        else
          $window.location.href = '/trained.html';
      }else $scope.errorCallback();
    }

    $scope.errorCallback = function(response){
      $scope.loginFailed = true;
      $scope.loginForm.password.$setPristine();
    }
  }])


  .controller("MovieListController", ['$scope', 'movieFactory', function($scope, movieFactory){
    $scope.movies = movieFactory.getMovies();
  }])

  .controller("BookSeatsController", ['$scope', '$routeParams', 'movieFactory', function($scope, $routeParams, movieFactory){
    $scope.movie = movieFactory.getMovie(parseInt($routeParams.id,10));
    $scope.getSeatNumber = function(number){ return new Array(number);}
  }])
	.factory('movieFactory', function(){

		var moviefac = {};

		var movies = [{
                      	"_id" : 0,
                      	"title" : "Legend Of Tarzan",
                      	"description" : "Tarzan, having acclimated to life in London, is called back to his former home in the jungle to investigate the activities at a mining encampment.",
                      	"rating" : 4.6,
                        "seats-available" : 60,
                      	"posterUrl" : "https://image.tmdb.org/t/p/w300_and_h450_bestv2/6FxOPJ9Ysilpq0IgkrMJ7PubFhq.jpg"
                      },
                      {
                      	"_id" : 1,
                      	"title" : "Independence Day: Resurgence",
                      	"description" : "We always knew they were coming back. Using recovered alien technology, the nations of Earth have collaborated on an immense defense program to protect the planet. But nothing can prepare us for the aliens’ advanced and unprecedented force.",
                      	"rating" : 4.7,
                        "seats-available" : 45,
                      	"posterUrl" : "https://image.tmdb.org/t/p/w300_and_h450_bestv2/9KQX22BeFzuNM66pBA6JbiaJ7Mi.jpg"
                      },
                      {
                      	"_id" : 2,
                      	"title" : "Star Trek Beyond",
                      	"description" : "The USS Enterprise crew explores the furthest reaches of uncharted space, where they encounter a mysterious new enemy who puts them and everything the Federation stands for to the test.",
                      	"rating" : 5.9,
                        "seats-available" : 90,
                      	"posterUrl" : "https://image.tmdb.org/t/p/w300_and_h450_bestv2/ghL4ub6vwbYShlqCFHpoIRwx2sm.jpg"
                      },
                      {
                      	"_id" : 3,
                      	"title" : "Finding Dory",
                      	"description" : "Finding Dory reunites Dory with friends Nemo and Marlin on a search for answers about her past. What can she remember? Who are her parents? And where did she learn to speak Whale?",
                      	"rating" : 6.4,
                        "seats-available" : 98,
                      	"posterUrl" : "https://image.tmdb.org/t/p/w300_and_h450_bestv2/z09QAf8WbZncbitewNk6lKYMZsh.jpg"
                      },
                      {
                      	"_id" : 4,
                      	"title" : "Ice Age: Collision Course",
                      	"description" : "Set after the events of Continental Drift, Scrats epic pursuit of his elusive acorn catapults him outside of Earth, where he accidentally sets off a series of cosmic events that transform and threaten the planet.…",
                      	"rating" : 5.3,
                        "seats-available" : 120,
                      	"posterUrl" : "https://image.tmdb.org/t/p/w300_and_h450_bestv2/1Rs4oQs5wONsWAZoqvOo1x9CjPC.jpg"
                      },
                      {
                      	"_id" : 5,
                      	"title" : "Ghostbusters",
                      	"description" : "Following a ghost invasion of Manhattan, paranormal enthusiasts Erin Gilbert and Abby Yates, nuclear engineer Jillian Holtzmann, and subway worker Patty Tolan band together to stop the otherworldly threat.",
                      	"rating" : 5.2,
                        "seats-available" : 81,
                      	"posterUrl" : "https://image.tmdb.org/t/p/w300_and_h450_bestv2/4qnJ1hsMADxzwnOmnwjZTNp0rKT.jpg"
                      }
                    ];


			moviefac.getMovies = function(){
				return movies;
			}

			moviefac.getMovie = function(index){
				return movies[index];
			}
			return moviefac;
	})
