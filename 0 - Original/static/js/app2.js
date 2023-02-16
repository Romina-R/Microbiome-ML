Plotly.d3.csv('https://raw.githubusercontent.com/pGuillergan/Group-Project-3/master/2%20-%20Karina/PCA_with_metadata.csv', function(error, rows){
    console.log(rows)
    function unpack(rows, key) {
        return rows.map(function(row) { return row[key]; });
    }
    colors = []
    for (i=0; i < unpack(rows, 'sample_type').length; i++) {
      if (unpack(rows, 'sample_type')[i] == "skin") {
        colors.push(0)
      } else if (unpack(rows, 'sample_type')[i] == "stool") {
        colors.push(0.5)
      } else if (unpack(rows, 'sample_type')[i] == "saliva") {
        colors.push(1)
      }
    }

    var pl_colorscale=[
        [0.0, '#19d3f3'],
        [0.333, '#19d3f3'],
        [0.333, '#a262a9'],
        [0.666, '#a262a9'],
        [0.666, '#636efa'],
        [1, '#636efa']
]

    var data = [{
        x: unpack(rows, 'PCoA-1'),
        y: unpack(rows, 'PCoA-2'),
        z: unpack(rows, 'PCoA-3'),
        mode: 'markers',
        type: 'scatter3d',
        marker: {
          color: colors,
        //   color: unpack(rows, 'sample_type'),
          colorscale:pl_colorscale,
            // color: 'rgb(23, 190, 207)',
          size: 2
        }
    },{
        alphahull: 7,
        opacity: 0.1,
        type: 'mesh3d',
        x: unpack(rows, 'PCoA-1'),
        y: unpack(rows, 'PCoA-2'),
        z: unpack(rows, 'PCoA-3')
    }];

    var layout = {
        // legend: {

        // },
        autosize: true,
        height: 800,
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        scene: {
            aspectratio: {
                x: 1,
                y: 1,
                z: 1
            },
            camera: {
                center: {
                    x: 0,
                    y: 0,
                    z: 0
                },
                eye: {
                    x: 1.25,
                    y: 1.25,
                    z: 1.25
                },
                up: {
                    x: 0,
                    y: 0,
                    z: 1
                }
            },
            xaxis: {
                type: 'linear',
                title: "PCoA-1",
                zeroline: false
            },
            yaxis: {
                type: 'linear',
                title: "PCoA-2",
                zeroline: false
            },
            zaxis: {
                type: 'linear',
                title: "PCoA-3",
                zeroline: false
            }
        },
        title: 'PCA clustering',
        width: 900
    };

    Plotly.newPlot('myDiv_1', data, layout);

});