// first bar chart

var trace1 = {
    x: ['skin', 'saliva', 'stool'],
    y: [561.04, 176.50, 360.23],
    name: 'Has Dog',
    type: 'bar'
  };
  
  var trace2 = {
    x: ['skin', 'saliva', 'stool'],
    y: [453.19, 190.69, 336.64],
    name: 'Does not have Dog',
    type: 'bar'
  };
  
  var data = [trace1, trace2];
  
  var layout = {
      barmode: 'group',
      xaxis: {
        title: "Sample Type",
      },
      yaxis: {
        title: "Microbial Diversity",
      },
      title: 'Human with Dog'
    };
  
///second one
    var trace3 = {
        x: ['skin', 'saliva', 'stool'],
        y: [1024.25, 217.81, 141.65],
        name: 'Dog',
        type: 'bar'
      };
      
      var trace4 = {
        x: ['skin', 'saliva', 'stool'],
        y: [488.87, 186.07, 344.60],
        name: 'Human',
        type: 'bar'
      };
      
      var data2 = [trace3, trace4];
      
      var layout = {
          barmode: 'group',
          xaxis: {
            title: "Sample Type",
          },
          yaxis: {
            title: "Microbial Diversity",
          },
          title: 'Human vs Dog'
        };
      
    var data_bothPlots = [data, data2]
    //   Plotly.newPlot('myDiv_bar_DOGvHu', data_bothPlots, layout);


  Plotly.newPlot('myDiv', data_bothPlots, layout);