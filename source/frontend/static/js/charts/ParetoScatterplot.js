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
     * @param dataset
     * @param style Various style settings (chart width/height, colors, ...). Arbitrary format, has to be parsed indivdually
     * by concrete classes.
     * @param parentDivID
     */
    constructor(name, panel, attributes, dataset, style, parentDivID)
    {
        super(name, panel, attributes, dataset, style, parentDivID);

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
        this._cf_chart.render();
    }

    constructCFChart()
    {
        this._cf_chart = dc.scatterPlot("#" + this._target, this._dataset);

        // Create shorthand references.
        let instance    = this;
        let extrema     = this._dataset._cf_extrema;
        let dimensions  = this._dataset._cf_dimensions;
        let key         = this._axes_attributes.x + ":" + this._axes_attributes.y;

        // Configure chart.
        this._cf_chart
            .height(instance._style.height)
            .width(instance._style.width)
            .useCanvas(true)
            .x(d3.scale.linear().domain(
                [extrema[instance._axes_attributes.x].min, extrema[instance._axes_attributes.x].max])
            )
            .y(d3.scale.linear().domain(
                [extrema[instance._axes_attributes.y].min, extrema[instance._axes_attributes.y].max])
            )
            .xAxisLabel(instance._style.showAxisLabels ? instance._axes_attributes.x : null)
            .yAxisLabel(instance._style.showAxisLabels ? instance._axes_attributes.y : null)
            .clipPadding(0)
            .renderHorizontalGridLines(true)
            .dimension(dimensions[key])
            .group(this._dataset.cf_groups[key])
            .existenceAccessor(function(d) {
                return d.value.items.length > 0;
            })
            .mouseZoomable(true)
            .excludedSize(instance._style.excludedSymbolSize)
            .excludedOpacity(instance._style.excludedOpacity)
            .excludedColor(instance._style.excludedColor)
            .symbolSize(instance._style.symbolSize)
    //        .colorAccessor(function(d) {
    //            return d.key[2];
    //        })
    //        .colors(scatterplotColors)
            .keyAccessor(function(d) {
                return d.key[0];
             })
            // Filter on end of brushing action, not meanwhile (performance suffers otherwise).
            .filterOnBrushEnd(true)
            .excludedOpacity(instance._style.excludedOpacity)
            .mouseZoomable(true)
            .margins({top: 0, right: 5, bottom: 25, left: 20});

        // Set number of ticks.
        this._cf_chart.yAxis().ticks(instance._style.numberOfTicks.y);
        this._cf_chart.xAxis().ticks(instance._style.numberOfTicks.x);

        // this._cf_chart.on('pretransition', function(chart) {
        //     instance._cf_chart.selectAll('g.row').on('mouseover', function() {
        //     });
        // });
    }
}