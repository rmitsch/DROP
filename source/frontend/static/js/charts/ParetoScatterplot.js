import Scatterplot from "./Scatterplot.js";

/**
 * Scatterplots with dots connected by specified degree of freedom.
 */
export default class ParetoScatterplot extends Scatterplot
{
    /**
     * Instantiates new ParetoScatterplot.
     * @param name
     * @param panel
     * @param attributes Attributes to be considered in plot. Has to be of length 2. First argument is projected onto
     * x-axis, second to y-axis. Attributes can contain one hyperparameter and one objective or two objectives (might
     * produce unspecified behaviour if handled otherwise; currently not checked in code).
     * @param crossfilter
     */
    constructor(name, panel, attributes, crossfilter)
    {
        super(name, panel, attributes, crossfilter);

        // Update involved CSS classes.
        $("#" + this._target).addClass("pareto-scatterplot");
    }

    /**
     * Connects model parametrizations by specified attribute name. Used to e. g. draw pareto fronts.
     * @param hyperparameter Name of attribute to be used to connect models. Default value 'native' connects models with
     * diverging values for attribute on x-axis, but constant for all hyperparameter otherwise. Note that for
     * hyperparameter-objective plots all attributes other than native are ignored. Objective-objective plots accept all
     * available hyperparameters.
     */
    connectBy(hyperparameter = 'native')
    {
        throw new Error("ParetoScatterplot.connectBy(): Not implemented yet.");
    }

    render()
    {
        throw new Error("ParetoScatterplot.render(): Not implemented yet.");
    }

    constructCFChart()
    {
        this._cf_chart = dc.scatterPlot("#" + this._target);

        let instance = this;
        this._cf_chart
            .height(137)
            .x(d3.scale.linear().domain([instance._cf_extrema.x.min, instance._cf_extrema.x.max]))
            .y(d3.scale.linear().domain([instance._cf_extrema.y.min, instance._cf_extrema.y.max]))
            .xAxisLabel(instance._axes_attributes.x)
            .yAxisLabel(instance._axes_attributes.y)
            .clipPadding(0)
            .renderHorizontalGridLines(true)
            .dimension(instance._cf_dimensions.total)
            .group(instance._cf_groups.total)
            .existenceAccessor(function(d) {
                return d.value.items.length > 0;
            })
            .symbolSize(2)
    //        .colorAccessor(function(d) {
    //            return d.key[2];
    //        })
    //        .colors(scatterplotColors)
            .keyAccessor(function(d) {
                return d.key[0];
             })
            .excludedOpacity(0.65)
            .mouseZoomable(true)
            .margins({top: 5, right: 0, bottom: 50, left: 45});

        this._cf_chart.yAxis().ticks(4);
        this._cf_chart.xAxis().ticks(4);

        this._cf_chart.on('pretransition', function(chart) {
            instance._cf_chart.selectAll('g.row')
                .on('mouseover', console.log("over"));
        });
    }
}