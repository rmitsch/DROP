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

        this._genenerate_sampleVarianceBySample_histogram(dcGroupName);

        // -----------------------------------
        // 2. Generate heatmap.
        // -----------------------------------

        // this._generateDissonanceHeatmap(dcGroupName);

        // -----------------------------------
        // 3. Generate vertical (k-neighbour-
        // hood) histogram.
        // -----------------------------------

        this._generate_sampleVarianceByK_histogram(dcGroupName);
    }

    render()
    {
        let newHeight   = null;
        let newWidth    = null;

        // -------------------------------
        // Render horizontal histogram.
        // -------------------------------

        this._sampleVarianceBySampleHistogram.render();

        // -------------------------------
        // Render vertical histogram.
        // -------------------------------

        // Has to be drawn with updated height value.
        newHeight = $("#" + this._panel._target).height() - 55;
        this._sampleVarianceByKHistogram.width(newHeight);
        $("#" + this._divStructure.kHistogramDivID).css({
            "top": (
                this._sampleVarianceByKHistogram.width() / 2 +
                // Additional margin to align with heatmap.
                9
            ) + "px",
            "left": -(
                this._sampleVarianceByKHistogram.width() / 2 -
                this._sampleVarianceByKHistogram.margins().top -
                this._sampleVarianceByKHistogram.margins().bottom -
                8
            ) + "px"
        });
        this._sampleVarianceByKHistogram.render();

        // -------------------------------
        // Render heatmap.
        // -------------------------------

        // this._dissonanceHeatmap.width(
        //     this._sampleVarianceBySampleHistogram.width() -
        //     this._dissonanceHeatmap.margins().right -
        //     this._sampleVarianceBySampleHistogram.margins().right -
        //     8
        // );
        // this._dissonanceHeatmap.height(
        //     this._sampleVarianceByKHistogram.width() -
        //     this._sampleVarianceByKHistogram.margins().left -
        //     8
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
        let axesAttributes = {
            x: "r_nx",
            y: "b_nx"
        };
        let key         = axesAttributes.x + ":" + axesAttributes.y;

        var heatColorMapping = d3.scale.linear()
            .domain([0, 1])
            .range(["red", "#fff"]);

        // Configure chart.
        this._dissonanceHeatmap
            .height(300)
            .width(300)
            .dimension(dimensions[key])
            .group(dataset.cf_groups[key])
           .colorAccessor(function(d) {
               return Math.random();
           })
            .colors(heatColorMapping)
            .keyAccessor(function(d) {
                return d.key[0];
             })
            .valueAccessor(function(d) {
                return d.key[1];
             })
            .colsLabel(function(d) { return ""; })
            .rowsLabel(function(d) { return ""; })
            .margins({top: 0, right: 20, bottom: 0, left: 0});

        this._dissonanceHeatmap.xBorderRadius(0);
        this._dissonanceHeatmap.yBorderRadius(0);
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
        this._sampleVarianceBySampleHistogram = dc.barChart(
            "#" + this._divStructure.sampleHistogramDivID,
            dcGroupName
        );

        // Use arbitrary axis attribute for prototype purposes.
        let xAttribute = "sample_id";
        // Create shorthand references.
        let yAttribute = "sample_id#measure";

        // Configure chart.
        this._sampleVarianceBySampleHistogram
            .height(40)
            // 0.8: Relative width of parent div.
            .width($("#" + this._target).width())
            .valueAccessor( function(d) { return d.value; } )
            .elasticY(false)
            .x(d3.scale.linear().domain([0, extrema[xAttribute].max]))
            .y(d3.scale.linear().domain([0, 1]))
            .brushOn(true)
            .filterOnBrushEnd(true)
            .dimension(dimensions[xAttribute])
            .group(dataset._cf_groups[yAttribute])
            // See for info on ordering (not sure if it works as specified here):
            // https://stackoverflow.com/questions/25204782/sorting-ordering-the-bars-in-a-bar-chart-by-the-bar-values-with-dc-js
            .ordering(function(d) { return d[xAttribute]; })
            .renderHorizontalGridLines(true)
            .margins({top: 5, right: 5, bottom: 5, left: 30})
            .gap(0);

        // Set number of ticks.
        this._sampleVarianceBySampleHistogram.yAxis().ticks(1);
        this._sampleVarianceBySampleHistogram.xAxis().ticks(0);
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
        this._sampleVarianceByKHistogram = dc.barChart(
            "#" + this._divStructure.kHistogramDivID,
            dcGroupName
        );

        // Define which attributes to use.
        let xAttribute = "model_id";
        let yAttribute = "model_id#measure";

        // Configure chart.
        this._sampleVarianceByKHistogram
            .height(40)
            .width($("#" + this._panel._target).height())
            .valueAccessor( function(d) { return d.value; } )
            .elasticY(false)
            .x(d3.scale.linear().domain([0, extrema[xAttribute].max]))
            .y(d3.scale.linear().domain([0, 1]))
            .brushOn(true)
            .filterOnBrushEnd(true)
            .dimension(dimensions[xAttribute])
            .group(dataset._cf_groups[yAttribute])
            .renderHorizontalGridLines(true)
            .useRightYAxis(false)
            .yAxisLabel("")
            .margins({top: 5, right: 5, bottom: 5, left: 15})
            .gap(0);

        // Set number of ticks.
        this._sampleVarianceByKHistogram.yAxis().ticks(1);
        this._sampleVarianceByKHistogram.xAxis().ticks(0);
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

        let sampleHistogramDiv  = Utils.spawnChildDiv(this._target, null, "dissonance-variance horizontal");
        let heatmapDiv          = Utils.spawnChildDiv(this._target, null, "dissonance-heatmap");
        let kHistogramDiv       = Utils.spawnChildDiv(this._target, null, "dissonance-variance-chart vertical");

        return {
            sampleHistogramDivID: sampleHistogramDiv.id,
            heatmapDivID: heatmapDiv.id,
            kHistogramDivID: kHistogramDiv.id
        };
    }
}