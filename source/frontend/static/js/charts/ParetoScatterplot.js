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
        // Use operator's target ID as group name.
        this._cf_chart = dc.scatterPlot(
            "#" + this._target,
            this._panel._operator._target,
            this._dataset,
            this._axes_attributes.x
        );

        // Create shorthand references.
        let instance    = this;
        let extrema     = this._dataset._cf_extrema;
        let dimensions  = this._dataset._cf_dimensions;
        let key         = this._axes_attributes.x + ":" + this._axes_attributes.y;

        // NEXT:
        //     - use xAxis().tickValues to set ordinal ticks for categorical chart (fix space problem later)
        //     - add table
        //     - fix bug: non-selected point in currently active chart shouldn't be displayed'
        //     - improve: show lines between inactive points in series in gray instead not at all (solution similart to point above)
        //     - add scrollpanes to layout

        // Configure chart.
        this._cf_chart
            .height(instance._style.height)
            .width(instance._style.width)
            .useCanvas(true)
            .x(d3.scale.linear().domain(
                [extrema[instance._axes_attributes.x].min, extrema[instance._axes_attributes.x].max]
            ))
            .y(d3.scale.linear().domain(
                [extrema[instance._axes_attributes.y].min, extrema[instance._axes_attributes.y].max]
            ))

            .xAxisLabel(instance._style.showAxisLabels ? instance._axes_attributes.x : null)
            .yAxisLabel(instance._style.showAxisLabels ? instance._axes_attributes.y : null)
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
            .margins({top: 0, right: 0, bottom: 25, left: 25});

        // Set number of ticks for y-axis.
        this._cf_chart.yAxis().ticks(instance._style.numberOfTicks.y);
        this._cf_chart.xAxis().ticks(instance._style.numberOfTicks.x);

        // If this x-axis hosts categorical argument: Print categorical representations of numerical values.
        if (this._axes_attributes.x.indexOf("*") !== -1) {
            // Get original name by removing suffix "*" from attribute name.
            let originalAttributeName = instance._axes_attributes.x.slice(0, -1);

            // Overwrite number of ticks with number of possible categorical values.
            this._cf_chart.xAxis().ticks(
                Object.keys(this._dataset.numericalToCategoricalValues[originalAttributeName]).length
            );

            // Use .tickFormat to convert numerical to original categorical representations.
            this._cf_chart.xAxis().tickFormat(function (tickValue) {
                // Print original categorical for this numerical representation.
                return tickValue in instance._dataset.numericalToCategoricalValues[originalAttributeName] ?
                        instance._dataset.numericalToCategoricalValues[originalAttributeName][tickValue] : "";
            });
        }
    }
}