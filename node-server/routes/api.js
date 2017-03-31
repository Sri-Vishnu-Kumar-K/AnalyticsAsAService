var express = require('express');
var router = express.Router();
var formidable = require('formidable');
var fs = require('fs');
var util = require('util');
var sh = require('shelljs');
var request = require('request');
var mongoose = require('mongoose');
var mongodb = require('mongodb');
var DB_URI = 'mongodb://localhost:27017/AaaS';

//var MongoClient = mongodb.MongoClient;
mongoose.connect(DB_URI);

var User1 = mongoose.model('User',{username:String, password:String});
var loggedUser = "";
var User = {dataPath:""};
router.get('/', function(req, res, next){
	res.sendFile(__dirname + "/../public/index.html");
})

// router.get('/learn', function(req, res, next){
// 	res.sendFile(__dirname + "/../public/learn.html");
// })

router.post('/login',function(req, res, next){
	console.log("Query is : " + req.body.username);
	User1.findOne({username:req.body.username}, function(err, userObj){
		if (err) {
    console.log(err);
  	} else if (userObj) {
    	console.log('Found:', userObj);
			req.session.user = userObj;
  	} else {
    	console.log('User not found!');
  	}
		console.log("Over!")
		var failure = {
			status: false
		}

		var success = {
			status: true,
			pristine: true
		}

		loggedUser = req.session.user.username;
		//req.session.user = req.body;

		console.log("Username is ", loggedUser);
		req.body.username = req.session.user.username;
		var sendingData = {
			userName: req.body.username
		}
		var requestUrl = 'http://127.0.0.1:8000/Engine/dirExists/?data=' + JSON.stringify(sendingData);
		request(requestUrl, function(error, response, body) {
			if( !error && response.statusCode == 200) {
				console.log("Call done!");
			}

			//console.log("/dirExists response: ", response);
			// response = JSON.stringify(response);
			// res.json(response);
		})
		console.log("Cookie Value /login: ", req.session.user);
		if(req.body.username)
			res.json(success);
	});
})


router.post('/upload', function(req,res,next){
	var form = new formidable.IncomingForm();
	var data = null;
	var failure = {status : false}
	var success = {status : true}

	if(loggedUser == "") loggedUser = 'SBI';

	form.parse(req, function(err, fields, files){
		console.log(files.file.path)
		data = fs.readFileSync(files.file.path);

		var idText = makeid();
		var filepath = sh.pwd()+"/datasets/" + idText + ".csv";

		fs.writeFile(filepath , data, function(err) {
			if(err) {
				console.log(err);
				res.json(failure);
			}
			/*User.findOne({username: loggedUser},function(err, user){
				if(err) console.log(err);
				// original: user.dataset = filepath;
				User.dataset = filepath;
				/*user.save(function(err) {
					if (err) throw err;

					console.log('User successfully updated!');
				});
			})*/
			User.dataset = filepath;
			console.log("Cookie value /upload: ", req.session.user);
			res.json(success);
			console.log("The file was saved!", User.dataset);
		});
	})
})

router.get("/get-file-path", function(req, res, next){
	console.log("get-file-path");
	if(loggedUser == "") loggedUser = 'SBI';
	/*User.findOne({username: loggedUser},function(err, user){
				if(err) {
					console.log(err);
					res.json({});
				}
				var data = {
					// filepath: User.dataset
				}
				original: var data = {
					filepath : user.dataset
				}
	 });*/
			// original:	res.json(data);
			console.log("Cookie value /get-file-path: ", req.session.user);
			//console.log("The file path retrieved" + data.filepath);
			console.log(User.dataset);
			res.json({filepath: User.dataset });
})

router.post("/save-selected-feat", function(req, res, next){
	console.log("Cookie value /save-selected-feat: ", req.session.user);
	if(loggedUser == "") loggedUser = 'SBI';
	User.findOne({username: loggedUser},function(err, user){
				if(err) {
					console.log(err);
					res.json({});
				}
			//original	user.features = req.body.keys;
			User.features = req.body.keys;
			UserFeatures = req.body.keys;
			/*	user.save(function(err) {
					if (err) throw err;
					console.log(user.features)
					console.log('User successfully updated!');
				}); */

	})
})


router.get("/get-selected-feat", function(req, res, next){
	console.log("Cookie value /get-selected-feat: ", req.session.user);
	if(loggedUser == "") loggedUser = 'SBI';
	User.findOne({username: loggedUser},function(err, user){
				if(err) {
					console.log(err);
					res.json({});
				}
				var data = {
					keys : User.features// User.features user.features
				}
				res.json(data);
			console.log("The file path retrieved" + data.filepath);
	})
})

router.get("/get-user-name", function(req, res, next){
	var data = {
		userName: req.session.username
	};

	data = JSON.stringify(data);
	res.json(data);
})


router.post("/call-build-model", function(req, res, next){

	 var jsonData = JSON.parse(req.body.data);
	 console.log(jsonData);
	 var requestUrl = 'http://127.0.0.1:8000/Engine/buildModel' + jsonData.urlType + '/?data=';
	 console.log(requestUrl);
	 jsonData.urlData.userName = req.session.user.username;
	 requestUrl = requestUrl + JSON.stringify(jsonData.urlData);
	 console.log(requestUrl);
	 request(requestUrl, function (error, response, body) {
    if (!error && response.statusCode == 200) {
        console.log("Call done!");
     }

		 response = JSON.stringify(response);
		 res.json(response);

})
})

router.post("/call-run-model", function(req, res, next){

	var jsonData = JSON.parse(req.body.data);
	console.log(jsonData);
	jsonData.userName = req.session.user.username;
	var requestUrl = 'http://127.0.0.1:8000/Engine/runModel/?data=' + JSON.stringify(jsonData);
	console.log("/call-run-model requestUrl: ", requestUrl);
	request(requestUrl, function(error, response, body) {
		if( !error && response.statusCode == 200) {
			console.log("Call done!");
		}

		response = JSON.stringify(response);
		res.json(response);
	})
})

router.post("/call-get-columns", function(req, res, next){
	var jsonData = JSON.parse(req.body.data);
	console.log(jsonData);
	jsonData.userName = req.session.user.username;
	var requestUrl = 'http://127.0.0.1:8000/Engine/getColumns/?data=' + JSON.stringify(jsonData);
	console.log("call-get-columns request url = ", requestUrl);
	request(requestUrl, function(error, response, body) {
		if( !error && response.statusCode == 200) {
			console.log("Call done!");
		}

		response = JSON.stringify(response);
		res.json(response);
	})
})

router.post("/call-get-models", function(req, res, next){
	var jsonData = JSON.parse(req.body.data);
	console.log(jsonData);
	jsonData.userName = req.session.user.username;
	var requestUrl = 'http://127.0.0.1:8000/Engine/getModels/?data='+JSON.stringify(jsonData);
	request(requestUrl, function(error, response, body) {
		if( !error && response.statusCode == 200) {
			console.log("Call done!");
		}

		var responseToBeSent = response;
		responseToBeSent.userName = req.session.username;
		responseToBeSent = JSON.stringify(responseToBeSent);

		res.json(responseToBeSent);
	})
})

function makeid()
{
    var text = "";
    var possible = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

    for( var i=0; i < 10; i++ )
        text += possible.charAt(Math.floor(Math.random() * possible.length));

    return text;
}

module.exports = router;
