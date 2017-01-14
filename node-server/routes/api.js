var express = require('express');
var router = express.Router();
var formidable = require('formidable');
var fs = require('fs');
var util = require('util');

var mongoose = require('mongoose');
var DB_URI = 'mongodb://localhost/nautilus';
mongoose.connect(DB_URI);


var User = require("../models/user");

var loggedUser = "";

router.get('/', function(req, res, next){
	res.sendFile(__dirname + "/../public/index.html");
})

// router.get('/learn', function(req, res, next){
// 	res.sendFile(__dirname + "/../public/learn.html");
// })

router.post('/login',function(req, res, next){
	console.log("Query is : " + req.body.username);
	User.findOne({username: req.body.username}, 'username password pristine', function (err, user) {
		
		var failure = {
			status : false
		}
		
		var success = {
			status : true,
			pristine : user.pristine
		}
		
		if (err) res.json(failure);
		if(user.password === req.body.password){
			User.update({username: user.username}, {$set: {pristine: false}}, {}, function(err){});
			loggedUser = user.username;
			res.json(success);
		}
		else res.json(failure);
	})
	
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
		var filepath = "/home/sudar/Desktop/bears/nautilus/datasets/" + makeid() + ".csv";
		
		fs.writeFile(filepath , data, function(err) {
			if(err) {
				console.log(err);
				res.json(failure);
			}
			User.findOne({username: loggedUser},function(err, user){
				if(err) console.log(err);
				user.dataset = filepath;
				user.save(function(err) {
					if (err) throw err;
				
					console.log('User successfully updated!');
				});
			})
			res.json(success);
			console.log("The file was saved!");
		}); 
	})
})

router.get("/get-file-path", function(req, res, next){
	if(loggedUser == "") loggedUser = 'SBI';
	User.findOne({username: loggedUser},function(err, user){
				if(err) {
					console.log(err);
					res.json({});
				}
				var data = {
					filepath : user.dataset
				}
				
				res.json(data);
			console.log("The file path retrieved" + data.filepath);
	})
})

router.post("/save-selected-feat", function(req, res, next){
	if(loggedUser == "") loggedUser = 'SBI';
	User.findOne({username: loggedUser},function(err, user){
				if(err) {
					console.log(err);
					res.json({});
				}
				user.features = req.body.keys;
				user.save(function(err) {
					if (err) throw err;
					console.log(user.features)
					console.log('User successfully updated!');
				});
				
	})
})


router.get("/get-selected-feat", function(req, res, next){
	if(loggedUser == "") loggedUser = 'SBI';
	User.findOne({username: loggedUser},function(err, user){
				if(err) {
					console.log(err);
					res.json({});
				}
				var data = {
					keys : user.features
				}
				res.json(data);
			console.log("The file path retrieved" + data.filepath);
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