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

        // Generate div structure for child nodes.
        this._divStructure = this._createDivStructure();

        // Construct graph.
        this.constructCFChart();
    }

    constructCFChart()
    {
        // Use operator's target ID as group name.
        let dcGroupName = this._panel._operator._target;

        // -----------------------------------
        // 1. Generate horizontal (sample)
        // histogram.
        // -----------------------------------

        this._generateHorizontalHistogram(dcGroupName);

        // -----------------------------------
        // 2. Generate heatmap.
        // -----------------------------------

        this._generateDissonanceHeatmap(dcGroupName);

        // -----------------------------------
        // 3. Generate vertical (k-neighbour-
        // hood) histogram.
        // -----------------------------------

        this._generateVerticalHistogram(dcGroupName);
    }

    render()
    {
        let numCols     = this._dataset._binCounts.x;
        let numRows     = this._dataset._binCounts.y;
        // Use heatmap width and height as yard stick for histograms.
        let newHeight   = Math.floor(
            ($("#" + this._panel._target).height() * 0.9 - 55) / numRows
        ) * numRows;
        let newWidth    = Math.floor(
            $("#" + this._target).width() * 0.9 / numCols
        ) * numCols;
        console.log(newWidth);

        var data = this._dataset._cf_groups["samplesInModelsMeasure:sampleDRModelMeasure"].all();
        var ncols = d3.set(data.map(function(x) { return x.key[0]; })).size();
        var nrows = d3.set(data.map(function(x) { return x.key[1]; })).size();
        console.log(ncols + "; " + nrows);

        // -------------------------------
        // 1. Render horizontal histogram.
        // -------------------------------

        this._horizontalHistogram.width(
            newWidth +
            this._horizontalHistogram.margins().left +
            this._horizontalHistogram.margins().right
        );
        this._horizontalHistogram.render();

        // -------------------------------
        // 2. Render vertical histogram.
        // -------------------------------

        // Has to be drawn with updated height value.
        this._verticalHistogram.width(
            newHeight +
            this._verticalHistogram.margins().left +
            this._verticalHistogram.margins().right
        );

        $("#" + this._divStructure.verticalHistogramDivID).css({
            "top": (
                this._verticalHistogram.width() / 2 +
                // Additional margin to align with heatmap.
                8
            ) + "px",
            "left": -(
                this._verticalHistogram.width() / 2 -
                this._verticalHistogram.margins().top -
                this._verticalHistogram.margins().bottom -
                17
            ) + "px"
        });
        this._verticalHistogram.render();

        // -------------------------------
        // 3. Render heatmap.
        // -------------------------------

        this._dissonanceHeatmap.width(newWidth);
        this._dissonanceHeatmap.height(newHeight);
        this._dissonanceHeatmap.render();
    }

    /**
     * Generates dissonance heatmap.
     * @param dcGroupName
     * @private
     */
    _generateDissonanceHeatmap(dcGroupName)
    {
        // Use operator's target ID as group name.
        this._dissonanceHeatmap = dc.heatMap(
            "#" + this._divStructure.heatmapDivID,
            dcGroupName
        );

        // Create shorthand references.
        let dataset     = this._dataset;
        let extrema     = dataset._cf_extrema;
        let dimensions  = dataset._cf_dimensions;
        let attribute   = "samplesInModelsMeasure:sampleDRModelMeasure";

        // Configure chart.
        this._dissonanceHeatmap
            .height(300)
            .width(300)
            .dimension(dimensions[attribute])
            .group(dataset._cf_groups[attribute])
            .colorAccessor(function(d) {
                return d.value;
            })
            .colors(
                d3.scale
                    .linear()
                    .domain([0, extrema[attribute].max])
                    .range(["white", "blue"])
            )
            .keyAccessor(function(d) {
                return d.key[0];
             })
            .valueAccessor(function(d) {
                return d.key[1];
             })
            .title(function(d) {
                return "";
            })
            // Supress column/row label output.
            .colsLabel(function(d) { return ""; })
            .rowsLabel(function(d) { return ""; })
            .margins({top: 0, right: 20, bottom: 0, left: 0});

        // No rounded corners.
        this._dissonanceHeatmap.xBorderRadius(0);
        this._dissonanceHeatmap.yBorderRadius(0);
    }

    /**
     * Initializes horizontal histogram for sample variance per sample.
     * @param dcGroupName
     * @private
     */
    _generateHorizontalHistogram(dcGroupName)
    {
        // Create shorthand references.
        let dataset     = this._dataset;
        let extrema     = dataset._cf_extrema;
        let dimensions  = dataset._cf_dimensions;
        let xAttribute  = "measure";
        let yAttribute  = "samplesInModels#" + xAttribute;
        let binWidth    = dataset._binWidths[yAttribute];

        // Generate dc.js chart object.
        this._horizontalHistogram = dc.barChart(
            "#" + this._divStructure.horizontalHistogramDivID,
            dcGroupName
        );

        // Configure chart.
        this._horizontalHistogram
            .height(40)
            .width(Math.floor($("#" + this._target).width() / dataset._binCounts.x) * dataset._binCounts.x)
            .valueAccessor( function(d) { return d.value; } )
            .elasticY(false)
            .x(d3.scale.linear().domain([0, extrema[xAttribute].max]))
            .y(d3.scale.linear().domain([0, extrema[yAttribute].max]))
            .brushOn(true)
            .filterOnBrushEnd(true)
            .dimension(dimensions[xAttribute])
            .group(dataset._cf_groups[yAttribute])
            .margins({top: 5, right: 5, bottom: 5, left: 40})
            .gap(0);

        // Set bar width.
        this._horizontalHistogram.xUnits(dc.units.fp.precision(binWidth));
        // Set tick format on y-axis.
        this._horizontalHistogram.yAxis().tickFormat(d3.format('.3s'));
        // Set number of ticks.
        this._horizontalHistogram.yAxis().ticks(2);
        this._horizontalHistogram.xAxis().ticks(0);
    }

    /**
     * Initializes vertical histogram for sample variance per k-neighbourhood.
     * @param dcGroupName
     * @private
     */
    _generateVerticalHistogram(dcGroupName)
    {
        // Create shorthand references.
        let dataset     = this._dataset;
        let extrema     = dataset._cf_extrema;
        let dimensions  = dataset._cf_dimensions;
        let xAttribute  = this._dataset._supportedDRModelMeasure;
        let yAttribute  = "samplesInModels#" + xAttribute;
        let binWidth    = dataset._binWidths[yAttribute];

        // Generate dc.js chart object.
        this._verticalHistogram = dc.barChart(
            "#" + this._divStructure.verticalHistogramDivID,
            dcGroupName
        );

        // Configure chart.
        this._verticalHistogram
            .height(40)
            .width($("#" + this._panel._target).height())
            .valueAccessor( function(d) { return d.value; } )
            .elasticY(false)
            .x(d3.scale.linear().domain([0, extrema[xAttribute].max]))
            .y(d3.scale.linear().domain([0, extrema[yAttribute].max]))
            .brushOn(true)
            .filterOnBrushEnd(true)
            .dimension(dimensions[xAttribute])
            .group(dataset._cf_groups[yAttribute])
            .margins({top: 5, right: 5, bottom: 5, left: 35})
            .gap(0);

        // Set bar width.
        this._verticalHistogram.xUnits(dc.units.fp.precision(binWidth));
        // Set tick format on y-axis.
        this._verticalHistogram.yAxis().tickFormat(d3.format('.3s'));
        // Set number of ticks.
        this._verticalHistogram.yAxis().ticks(2);
        this._verticalHistogram.xAxis().ticks(0);
    }

     /**
     * Create (hardcoded) div structure for child nodes.
     * @returns {Object}
     */
    _createDivStructure()
    {
        // -----------------------------------
        // Create charts container.
        // -----------------------------------

        let sampleHistogramDiv  = Utils.spawnChildDiv(this._target, null, "dissonance-variance-chart horizontal");
        let heatmapDiv          = Utils.spawnChildDiv(this._target, null, "dissonance-heatmap");
        let kHistogramDiv       = Utils.spawnChildDiv(this._target, null, "dissonance-variance-chart vertical");

        return {
            horizontalHistogramDivID: sampleHistogramDiv.id,
            heatmapDivID: heatmapDiv.id,
            verticalHistogramDivID: kHistogramDiv.id
        };
    }
}