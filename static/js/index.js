const express = require('express');
const app = express();

app.use(express.static(__dirname + "/static"));

app.get("/", function(req,res){
   res.sendFile(__dirname + "/templates/index.html")
});

// app.post("/", function(req,res){
    
// })

app.get("/about", function(req,res){
    res.sendFile(__dirname + "/templates/about.html")
});

app.get("/team", function(req,res){
    res.sendFile(__dirname + "/templates/team.html")
});


app.listen(3000 , function(){
    console.log("Server started on port:3000")