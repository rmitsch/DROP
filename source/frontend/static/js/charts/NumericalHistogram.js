import Histogram from "./Histogram.js"

/**
 * Creates numerical histogram.
 */
export default class NumericalHistogram extends Histogram
{
    /**
     *
     * @param name
     * @param panel
     * @param attributes
     * @param dataset
     * @param style
     * @param parentDivID
     */
    constructor(name, panel, attributes, dataset, style, parentDivID)
    {
        super(name, panel, attributes, dataset, style, parentDivID);
    }

    render()
    {
        this._cf_chart.render();
    }

    constructCFChart()
    {
        this._cf_chart = dc.barChart("#" + this._target);

        // Create shorthand references.
        let instance    = this;
        let extrema     = this._dataset._cf_extrema;
        let dimensions  = this._dataset._cf_dimensions;
        let key         = this._axes_attributes.x + "#histogram";

        // Configure chart.
        this._cf_chart
            .height(instance._style.height)
            .width(instance._style.width)
            .valueAccessor( function(d) { return  d.value.count; } )
            .elasticY(true)
            .x(d3.scale.linear().domain(
                [extrema[instance._axes_attributes.x].min, extrema[instance._axes_attributes.x].max])
            )
            .brushOn(true)
            .dimension(dimensions[key])
            .group(this._dataset.cf_groups[key])
            .renderHorizontalGridLines(true)
            .margins({top: 0, right: 10, bottom: 25, left: 25});

        // Set number of ticks.
        this._cf_chart.yAxis().ticks(instance._style.numberOfTicks.y);
        this._cf_chart.xAxis().ticks(instance._style.numberOfTicks.x);

        // Set tick format.
        this._cf_chart.xAxis().tickFormat(d3.format(".1s"));

        // Update bin width.
        let binWidth = this._dataset._cf_intervals[this._axes_attributes.x] / this._dataset._binCount;
        this._cf_chart.xUnits(dc.units.fp.precision(binWidth * 1));
    }
}