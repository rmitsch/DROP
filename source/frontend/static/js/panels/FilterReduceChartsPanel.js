import Panel from "./Panel.js";
import uuidv4 from "../utils.js";

/**
 * Panel holding scatterplots and histograms in operator FilterReduce.
 */
export default class FilterReduceChartsPanel extends Panel
{
    /**
     * Constructs new FilterReduce charts panel.
     * @param name
     * @param operator
     * @param linked_crossfilter Reference to crossfilter instance. Might be null. If not null, specified crossfilter
     * instance is used (useful when multiple panels inside the same operator are supposed to operate on the same
     * dataset).
     */
    constructor(name, operator, linked_crossfilter)
    {
        super(name, operator, linked_crossfilter);

        // For all hyper-parameter/objective combinations: Construct one scatterplot.
        // Visual separation.
        // For all hyper-objective/objective combinations: Construct one scatterplot.



        // CONTINUE HERE: Draw scatterplots, test brushing/linking.
        // Also: Style rules how (dynamic IDs)? Adaptions in terms of width/height, placement etc. probably necessary.
        // ndx -> crossfilter
        let dim_components = this._crossfilter.dimension(function(d) { return d["n_components"]; });
        let dim_runtime = this._crossfilter.dimension(function(d) { return d["runtime"]; });

        let components_extrema   = {
            min: dim_components.bottom(1)[0]['n_components'] * 0.9,
            max: dim_components.top(1)[0]['n_components'] * 1.1
        };
        let runtime_extrema   = {
            min: dim_runtime.bottom(1)[0]['runtime'] * 0.9,
            max: dim_runtime.top(1)[0]['runtime'] * 1.1
        };


        let scatterchartDim = this._crossfilter.dimension(function (d) {
            return [d["n_components"], d["runtime"]];
        });

        let scatterchartGroup = scatterchartDim.group().reduce(
            function(elements, item) {
                elements.model_parametrizations.push(item);
                elements.count++;

                return elements;
            },
            function(elements, item) {
                elements.model_parametrizations.splice(elements.model_parametrizations.indexOf(item), 1);
                elements.count--;

                return elements;
            },
            function() {
                return {model_parametrizations: [], count: 0};
            }
        );

        // Create div structure for this chart.
        let div         = document.createElement('div');
        div.id          = uuidv4();
        div.className   = 'chart';
        $("#" + this._target).append(div);


        let plot = dc.scatterPlot("#" + div.id);

        // Configure transactions scatterplot.
        var scatterplotColors   = d3.scale.ordinal()
                                    .domain(["Income", "Expenses"])
                                    .range(["green", "red"]);
        plot
            .height(137)
            .x(d3.scale.linear().domain([components_extrema.min, components_extrema.max]))
            .y(d3.scale.linear().domain([runtime_extrema.min, runtime_extrema.max]))
            .xAxisLabel("n_components")
            .yAxisLabel("runtime")
            .clipPadding(0)
            .renderHorizontalGridLines(true)
            .dimension(scatterchartDim)
            .group(scatterchartGroup)
            .existenceAccessor(function(d) {
                return d.value.model_parametrizations.length > 0;
            })
            .symbolSize(2)
    //        .colorAccessor(function(d) {
    //            return d.key[2];
    //        })
            .keyAccessor(function(d) {
                return d.key[0];
             })
    //        .colors(scatterplotColors)
            .excludedOpacity(0.75)
            .mouseZoomable(true)
            .margins({top: 5, right: 0, bottom: 50, left: 45});

        plot.yAxis().ticks(4);
        plot.xAxis().ticks(4);

        plot.on('pretransition', function(chart) {
            plot.selectAll('g.row')
                .on('mouseover', console.log("over"));
        });


        // Render charts.
        dc.renderAll();

    }
}