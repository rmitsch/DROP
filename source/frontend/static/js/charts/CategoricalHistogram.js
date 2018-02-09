import Histogram from "./Histogram.js"

/**
 * Creates categorical histogram.
 */
export default class CategoricalHistogram extends Histogram
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
        let dimensions  = this._dataset._cf_dimensions;
        let key         = this._axes_attributes.x + "#histogram";

        // Configure chart.
        this._cf_chart
            .height(instance._style.height)
            .width(instance._style.width)

            .elasticY(true)
            .x(d3.scale.ordinal())
            .xUnits(dc.units.ordinal)
            .brushOn(true)
            .barPadding(0.1)
            .dimension(dimensions[key])
            .group(this._dataset.cf_groups[key])
            .renderHorizontalGridLines(true)
            .margins({top: 0, right: 10, bottom: 25, left: 25});

        // Set number of ticks (x-axis is ignored).
        this._cf_chart.yAxis().ticks(instance._style.numberOfTicks.y);
    }
}