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
        // Use operator's target ID as group name.
        this._cf_chart = dc.barChart("#" + this._target, this._panel._operator._target);

        // Create shorthand references.
        let instance    = this;
        let extrema     = this._dataset._cf_extrema;
        let intervals   = this._dataset._cf_intervals;
        let dimensions  = this._dataset._cf_dimensions;
        let key         = this._axes_attributes.x + "#histogram";

        // Configure chart.
        this._cf_chart
            .height(instance._style.height)
            .width(instance._style.width)
            .valueAccessor( function(d) { return  d.value.count; } )
            .elasticY(false)
            .x(d3.scale.linear().domain([
                extrema[instance._axes_attributes.x].min,
                extrema[instance._axes_attributes.x].max +
                // Add padding so that last bar is not cut off in the middle.
                intervals[instance._axes_attributes.x] * 0.1
            ]))
            .y(d3.scale.linear().domain([0, extrema[key].max]))
            .brushOn(true)
            // Filter on end of brushing action, not meanwhile (performance suffers otherwise).
            .filterOnBrushEnd(true)
            .dimension(dimensions[key])
            .group(this._dataset.cf_groups[key])
            .renderHorizontalGridLines(true)
            .margins({top: 0, right: 10, bottom: 25, left: 25})
            .gap(1);

        // Set number of ticks.
        this._cf_chart.yAxis().ticks(instance._style.numberOfTicks.y);
        this._cf_chart.xAxis().ticks(instance._style.numberOfTicks.x);

        // Update bin width.
        let binWidth = this._dataset._cf_intervals[this._axes_attributes.x] / this._dataset._binCount;
        this._cf_chart.xUnits(dc.units.fp.precision(binWidth));
    }

    highlight(id, source)
    {
        if (source !== this._target) {
            if (id !== null) {
                let value           = this._dataset.getDataByID(id)[this._axes_attributes.x];
                let xAttribute      = this._axes_attributes.x;

                this._cf_chart.selectAll('rect.bar').each(function(d){
                    if (value >= d.data.value.extrema[xAttribute].min && value <= d.data.value.extrema[xAttribute].max) {
                        d3.select(this).attr("fill", "red");
                    }
                });
            }

            // Reset all bars to default color.
            else {
                this._cf_chart.selectAll('rect.bar').each(function(d){
                    d3.select(this).attr("fill", "#1f77b4");
                });
            }
        }
    }
}