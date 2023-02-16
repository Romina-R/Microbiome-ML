function show_results(data){
	
  // @Romina, you're gonna edit this block of code for dogs and humans

  // THIS IS WHAT THE RESULTS LOOK LIKE IN DICTIONARY @ /dynamic_results/yes/skin/all
// json_data = {
// "logisticRegression": "17.0%",
// "logisticRegression_human": "21.0%",
// "naiveBayes_dog": "13.0",
// "naiveBayes_human": "29.0%",
// "randomForest": "17.0%",
// "randomForest_human": "39.0%"
// }

// dog results
	// var lr_str = '<h4>Logistic Regression: ' + data.logisticRegression +'</h4>'+'<br>'
  var rf_str = '<h4>Random Forest: ' + data.randomForest_dog +'</h4>'+'<br>'
  // var nb_str = '<h4>Naive Bayes: ' + data.naiveBayes_dog +'</h4>'+'<br>'
// human results
  // var lr_str_hu = '<h4>Logistic Regression: ' + data.logisticRegression_human +'</h4>'+'<br>'
  var rf_str_hu = '<h4>Random Forest: ' + data.randomForest_human +'</h4>'+'<br>'
  // var nb_str_hu = '<h4>Naive Bayes: ' + data.naiveBayes_human +'</h4>'+'<br>'

 // get element for humans
  document.getElementById("res_show_human").innerHTML=rf_str_hu;
  // get element for dogs
	document.getElementById("res_show_dog").innerHTML=rf_str;
	document.getElementById("res_jumbo").style.display = "block";
};

function loadJsonData(hd, st, hr) {
  // console.log("im in the loadData function!")
  url = '/dynamic_results/'+hd+'/'+st+'/'+hr
  console.log(url)

  d3.json(url).then((data) => {

    // console.log("im in the json data to do area!")
    console.log(data)
    // console.log(data.logisticRegression)
    // console.log(data.randomForest)
    // console.log(data.logisticRegression_human)
    show_results(data);
  });
}

$("#my_form").submit(function(e) {

  e.preventDefault();
  
  var have_dog = document.getElementById('have_dog').value;
  var sample_type = document.getElementById('sample_type').value;
  var human_role = document.getElementById('human_role').value;
  
  // console.log(have_dog);
  // console.log(sample_type);
  // console.log(human_role);

  loadJsonData(have_dog, sample_type, human_role)

  console.log("done with ML")
    
});

