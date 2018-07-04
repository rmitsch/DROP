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

        this._generateDissonanceHeatmap(dcGroupName);

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

        this._dissonanceHeatmap.width(this._sampleVarianceBySampleHistogram.width() - this._dissonanceHeatmap.margins().right);
        this._dissonanceHeatmap.height(this._sampleVarianceByKHistogram.width() - this._sampleVarianceByKHistogram.margins().left);
        this._dissonanceHeatmap.render();
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
        //     - Follow stackoverflow.com/questions/51122700/dc-js-rectangular-brush-for-heat-_generateDissonanceHeatmap on
        //       turning scatterplot into heatmap (separate class justified?).
        //     - Continue with other issues -
        //         * Sample interpolation,
        //         * design of other panels,
        //         * data generation,
        //         * UMAP instead of t-SNE,
        //         * using real instead of mocked data for model-sample dissonance and surrogate model,
        //         * implementing correlation overview for alpha-omega SSPs,
        //         * implementing hexagon-heatmaps (with area brush! -> how?) for omega-omega plots - no detail view.

        // Use operator's target ID as group name.
        this._dissonanceHeatmap = dc.scatterPlot(
            "#" + this._divStructure.heatmapDivID,
            dcGroupName,
            this._dataset,
            // Don't make use of SSP line functionality.
            null
        );

        // Create shorthand references.
        let dataset     = this._dataset;
        let extrema     = dataset._cf_extrema;
        let dimensions  = dataset._cf_dimensions;
        let axesAttributes = {
            x: "r_nx",
            y: "b_nx"
        };
        let key = axesAttributes.x + ":" + axesAttributes.y;

        // Configure chart.
        this._dissonanceHeatmap
            .height(300)
            .width(300)
            .useCanvas(true)
            .x(d3.scale.linear().domain(
                [extrema[axesAttributes.x].min, extrema[axesAttributes.x].max]
            ))
            .y(d3.scale.linear().domain(
                [extrema[axesAttributes.y].min, extrema[axesAttributes.y].max]
            ))
            .xAxisLabel(axesAttributes.x)
            .yAxisLabel(axesAttributes.y)
            .renderHorizontalGridLines(true)
            .dimension(dimensions[key])
            .group(dataset.cf_groups[key])
            .existenceAccessor(function(d) {
                console.log(d.value.items.length);
                return d.value.items.length > 0;
            })
            .excludedSize(0)
            .excludedColor("#fff")
            .excludedOpacity(0)
            .symbolSize(3)
    //        .colorAccessor(function(d) {
    //            return d.key[2];
    //        })
    //        .colors(scatterplotColors)
            .keyAccessor(function(d) {
                return d.key[0];
             })
            .useRightYAxis(true)
            // Filter on end of brushing action, not meanwhile (performance suffers otherwise).
            .filterOnBrushEnd(true)
            .mouseZoomable(false)
            .margins({top: 0, right: 35, bottom: 5, left: 0});


        // Set number of ticks for y-axis.
        this._dissonanceHeatmap.yAxis().ticks(5);
        this._dissonanceHeatmap.xAxis().ticks(3);
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
        let axesAttribute = "r_nx";

        // Create shorthand references.
        let key = axesAttribute + "#histogram";

        // Configure chart.
        this._sampleVarianceBySampleHistogram
            .height(40)
            // 0.8: Relative width of parent div.
            .width($("#" + this._target).width())
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
            .margins({top: 5, right: 5, bottom: 5, left: 30});

        // Set number of ticks.
        this._sampleVarianceBySampleHistogram.yAxis().ticks(1);
        this._sampleVarianceBySampleHistogram.xAxis().ticks(0);

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
        this._sampleVarianceByKHistogram = dc.barChart(
            "#" + this._divStructure.kHistogramDivID,
            dcGroupName
        );

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
            .useRightYAxis(false)
            .yAxisLabel("")
            .margins({top: 5, right: 5, bottom: 35, left: 15});

        // Set number of ticks.
        this._sampleVarianceByKHistogram.yAxis().ticks(1);
        this._sampleVarianceByKHistogram.xAxis().ticks(5);


        // Update bin width.
        let binWidth = dataset._cf_intervals[axesAttribute] / dataset._binCount;
        this._sampleVarianceByKHistogram.xUnits(dc.units.fp.precision(binWidth * 1));
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