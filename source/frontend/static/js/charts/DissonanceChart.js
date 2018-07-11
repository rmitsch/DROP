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

        // // -----------------------------------
        // // 2. Generate heatmap.
        // // -----------------------------------
        //
        // // this._generateDissonanceHeatmap(dcGroupName);

        // -----------------------------------
        // 3. Generate vertical (k-neighbour-
        // hood) histogram.
        // -----------------------------------

        this._generateVerticalHistogram(dcGroupName);
    }

    render()
    {
        let newHeight   = null;
        let newWidth    = null;

        // -------------------------------
        // 1. Render horizontal histogram.
        // -------------------------------

        this._horizontalHistogram.render();

        // -------------------------------
        // 2. Render vertical histogram.
        // -------------------------------

        // Has to be drawn with updated height value.
        newHeight = $("#" + this._panel._target).height() - 55;
        this._verticalHistogram.width(newHeight);
        $("#" + this._divStructure.verticalHistogramDivID).css({
            "top": (
                this._verticalHistogram.width() / 2 +
                // Additional margin to align with heatmap.
                9
            ) + "px",
            "left": -(
                this._verticalHistogram.width() / 2 -
                this._verticalHistogram.margins().top -
                this._verticalHistogram.margins().bottom -
                8
            ) + "px"
        });
        this._verticalHistogram.render();

        //
        // // -------------------------------
        // // 3. Render heatmap.
        // // -------------------------------
        //
        // this._dissonanceHeatmap.width(
        //     this._horizontalHistogram.width() -
        //     8
        // );
        //
        // newHeight = this._verticalHistogram.width() * 1 -
        //     8 +
        //     (67.6);
        // let bla = this._dataset._recordCounts["model_id"] / newHeight;
        // console.log(1 / bla);
        // this._dissonanceHeatmap.height(
        //     newHeight
        // );
        //
        // this._dissonanceHeatmap.render();
    }

    /**
     *
     * @param dcGroupName
     * @private
     */
    _generateDissonanceHeatmap(dcGroupName)
    {
        // Next steps:
        //     x Draw common scatterplot (with useCanvas = true and filterOnBrushEnd: true).
        //     - Continue with other issues -
        //         * Sample interpolation,
        //         * design of other panels,
        //         * data generation,
        //         * UMAP instead of t-SNE,
        //         * using real instead of mocked data for model-sample dissonance and surrogate model,
        //         * implementing correlation overview for alpha-omega SSPs,
        //         * implementing hexagon-heatmaps (with area brush! -> how?) for omega-omega plots - no detail view.

        // Use operator's target ID as group name.
        this._dissonanceHeatmap = dc.heatMap(
            "#" + this._divStructure.heatmapDivID,
            dcGroupName
        );

        // Create shorthand references.
        let dataset     = this._dataset;
        let extrema     = dataset._cf_extrema;
        let dimensions  = dataset._cf_dimensions;
        let dimName     = "sample_id:model_id";
        let groupName   = dimName + "#measure";

        // Configure chart.
        this._dissonanceHeatmap
            .height(300)
            .width(300)
            .dimension(dimensions[dimName])
            .group(dataset._cf_groups[groupName])
            .colorAccessor(function(d) {
                return d.value;
            })
            .colors(
                d3.scale
                    .linear()
                    .domain([0, 1])
                    .range(["white", "red"])
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
            // Surpress column/row label output.
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
            .width($("#" + this._target).width())
            .valueAccessor( function(d) { return d.value; } )
            .elasticY(false)
            .x(d3.scale.linear().domain([0, 1]))
            .y(d3.scale.linear().domain([0, extrema[yAttribute].max]))
            .brushOn(true)
            .filterOnBrushEnd(true)
            .dimension(dimensions[xAttribute])
            .group(dataset._cf_groups[yAttribute])
            .margins({top: 5, right: 5, bottom: 5, left: 35})
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
            .x(d3.scale.linear().domain([0, 1]))
            .y(d3.scale.linear().domain([0, extrema[yAttribute].max]))
            .brushOn(true)
            .filterOnBrushEnd(true)
            .dimension(dimensions[xAttribute])
            .group(dataset._cf_groups[yAttribute])
            .margins({top: 5, right: 5, bottom: 5, left: 35})
            .gap(0);

        console.log(dataset._cf_groups[yAttribute].top(Infinity))
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