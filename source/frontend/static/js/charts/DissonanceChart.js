import Chart from "./Chart.js";
import Utils from "../Utils.js"


/**
 * Creates chart for model dissonance.
 * Creates a heatmap with adjuct histograms for sample variance per (1) samples and (2) k-neighbourhood.
 * Heatmap is of samples x k-neighbourhood.
 */
export default class DissonanceChart extends Chart
{
    /**
     *
     * @param name
     * @param panel
     * @param attributes Ignored.
     * @param dataset
     * @param style
     * @param parentDivID
     */
    constructor(name, panel, attributes, dataset, style, parentDivID)
    {
        super(name, panel, attributes, dataset, style, parentDivID);

        // Construct graph.
        this.constructCFChart();
    }

    constructCFChart()
    {
        // Create shorthand references.
        let dataset     = this._dataset;
        let extrema     = dataset._cf_extrema;
        let dimensions  = dataset._cf_dimensions;
        // Use operator's target ID as group name.
        let dcGroupName = this._panel._operator._target;

        // -----------------------------------
        // 1. Generate horizontal (sample)
        // histogram.
        // -----------------------------------

        this._genenerate_sampleVarianceBySample_histogram(dcGroupName);

        // -----------------------------------
        // 2. Generate heatmap.
        // -----------------------------------

        this._generateDissonanceHeatmap(dcGroupName);

        // -----------------------------------
        // 3. Generate vertical (k-neighbour-
        // hood) histogram.
        // -----------------------------------

        this._generate_sampleVarianceByK_histogram(dcGroupName);
    }

    render()
    {
        this._sampleVarianceBySampleHistogram.render();

        // Has to be drawn with updated height value.
        let newHeight = $("#" + this._panel._target).height() - this._sampleVarianceBySampleHistogram.height();
        let newOffset = (
            newHeight / 4 +
            this._sampleVarianceBySampleHistogram.height() +
            this._sampleVarianceBySampleHistogram.margins().left -
            this._sampleVarianceBySampleHistogram.margins().right
        );
        this._sampleVarianceByKHistogram.width(newHeight);
        $("#" + this._panel._divStructure.kHistogramDivID).css({
            "margin-top": newOffset + "px",
            "margin-right": -(
                newOffset +
                this._sampleVarianceByKHistogram.height() -
                this._sampleVarianceByKHistogram.margins().top -
                this._sampleVarianceByKHistogram.margins().bottom +
                3
            ) + "px"
        });
        this._sampleVarianceByKHistogram.render();
    }

    /**
     *
     * @param dcGroupName
     * @private
     */
    _generateDissonanceHeatmap(dcGroupName)
    {

    }

    /**
     * Initializes horizontal histogram for sample variance per sample.
     * @param dcGroupName
     * @private
     */
    _genenerate_sampleVarianceBySample_histogram(dcGroupName)
    {
        // Create shorthand references.
        let dataset     = this._dataset;
        let extrema     = dataset._cf_extrema;
        let dimensions  = dataset._cf_dimensions;

        // Generate dc.js chart object.
        this._sampleVarianceBySampleHistogram = dc.barChart("#" + this._panel._divStructure.sampleHistogramDivID, dcGroupName);

        // Use arbitrary axis attribute for prototype purposes.
        let axesAttribute = "r_nx";

        // Create shorthand references.
        let key = axesAttribute + "#histogram";

        // Configure chart.
        this._sampleVarianceBySampleHistogram
            .height(40)
            // 0.8: Relative width of parent div.
            .width($("#" + this._panel._divStructure.chartsContainerDivID).width())
            .valueAccessor( function(d) { return d.value.count; } )
            .elasticY(false)
            .x(d3.scale.linear().domain(
                [extrema[axesAttribute].min, extrema[axesAttribute].max])
            )
            .y(d3.scale.linear().domain([0, extrema[key].max]))
            .brushOn(true)
            // Filter on end of brushing action, not meanwhile (performance suffers otherwise).
            .filterOnBrushEnd(true)
            .dimension(dimensions[key])
            .group(dataset.cf_groups[key])
            .renderHorizontalGridLines(true)
            .margins({top: 5, right: 10, bottom: 5, left: 25});

        // Set number of ticks.
        this._sampleVarianceBySampleHistogram.yAxis().ticks(1);
        this._sampleVarianceBySampleHistogram.xAxis().ticks(0);

        // Set tick format.
        this._sampleVarianceBySampleHistogram.xAxis().tickFormat(d3.format(".1s"));

        // Update bin width.
        let binWidth = dataset._cf_intervals[axesAttribute] / dataset._binCount;
        this._sampleVarianceBySampleHistogram.xUnits(dc.units.fp.precision(binWidth * 1));
    }

    /**
     * Initializes vertical histogram for sample variance per k-neighbourhood.
     * @param dcGroupName
     * @private
     */
    _generate_sampleVarianceByK_histogram(dcGroupName)
    {
        // Create shorthand references.
        let dataset     = this._dataset;
        let extrema     = dataset._cf_extrema;
        let dimensions  = dataset._cf_dimensions;

        // Generate dc.js chart object.
        this._sampleVarianceByKHistogram = dc.barChart("#" + this._panel._divStructure.kHistogramDivID, dcGroupName);

        // Use arbitrary axis attribute for prototype purposes.
        let axesAttribute = "b_nx";

        // Create shorthand references.
        let key = axesAttribute + "#histogram";

        // Configure chart.
        this._sampleVarianceByKHistogram
            .height(40)
            .width($("#" + this._panel._target).height())
            .valueAccessor( function(d) { return d.value.count; } )
            .elasticY(false)
            .x(d3.scale.linear().domain(
                [extrema[axesAttribute].min, extrema[axesAttribute].max])
            )
            .y(d3.scale.linear().domain([0, extrema[key].max]))
            .brushOn(true)
            // Filter on end of brushing action, not meanwhile (performance suffers otherwise).
            .filterOnBrushEnd(true)
            .dimension(dimensions[key])
            .group(dataset.cf_groups[key])
            .renderHorizontalGridLines(true)
            .useRightYAxis(true)
            .margins({top: 5, right: 25, bottom: 5, left: 10});

        // Set number of ticks.
        this._sampleVarianceByKHistogram.yAxis().ticks(1);
        this._sampleVarianceByKHistogram.xAxis().ticks(0);

        // Set tick format.
        this._sampleVarianceByKHistogram.xAxis().tickFormat(d3.format(".1s"));

        // Update bin width.
        let binWidth = dataset._cf_intervals[axesAttribute] / dataset._binCount;
        this._sampleVarianceByKHistogram.xUnits(dc.units.fp.precision(binWidth * 1));
     }
}