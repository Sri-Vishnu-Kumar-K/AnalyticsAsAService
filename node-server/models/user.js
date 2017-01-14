var mongoose = require('mongoose');
var Schema = mongoose.Schema;

var userSchema = new Schema({
	username : { type: String, required: true, unique: true },
	password : { type: String, required: true},
	pristine: { type: Boolean, default : false },
	dataset: { type : String },
	features : [String]
});

var user = mongoose.model('User',userSchema);
module.exports = user;