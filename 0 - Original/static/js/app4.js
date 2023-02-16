var trace1 = {
    x: ['skin', 'saliva', 'stool'],
    y: [561.04, 176.50, 360.23],
    name: 'Dog owners',
    type: 'bar'
  };
  
  var trace2 = {
    x: ['skin', 'saliva', 'stool'],
    y: [453.19, 190.69, 336.64],
    name: 'Non-Dog Owners',
    type: 'bar'
  };
  
  var data = [trace1, trace2];
  
  var layout = {
      barmode: 'group',
      plot_bgcolor: 'rgba(0,0,0,0)',
      paper_bgcolor: 'rgba(0,0,0,0)',
      xaxis: {
        title: "Sample Type",
      },
      yaxis: {
        title: "Avg. Microbial Diversity",
      },
      title: 'Dog Owners vs. Non-dog Owners'
    };
  
  Plotly.newPlot('myDiv_3', data, layout);