import Panel from "./Panel.js";
import Utils from "../Utils.js";
import Dataset from "../Dataset.js";

/**
 * Panel holding elements for comparison of inter-model disagreement for datapoints
 * in selected model instances.
 */
export default class DissonancePanel extends Panel
{
    /**
     * Constructs new panel for charts for DissonanceOperator.
     * @param name
     * @param operator
     * @param parentDivID
     */
    constructor(name, operator, parentDivID)
    {
        super(name, operator, parentDivID);

        // Update involved CSS classes.
        $("#" + this._target).addClass("dissonance-panel");

        // Create div structure for child nodes.
        this._divStructure = this._createDivStructure();

        // Generate charts.
        this._generateCharts();
    }

    /**
     * Generates all chart objects. Does _not_ render them.
     */
    _generateCharts()
    {
        console.log("Generating DissonancePanel...");

        // Create shorthand references.
        let dataset     = this._operator._dataset;
        let extrema     = dataset._cf_extrema;
        let dimensions  = dataset._cf_dimensions;
        // Use operator's target ID as group name.
        let dcGroupName = this._operator._target;

        // -----------------------------------
        // 1. Generate horizontal (sample)
        // histogram.
        // -----------------------------------

        this._genenerate_sampleVarianceBySample_histogram();

        // -----------------------------------
        // 2. Generate heatmap.
        // -----------------------------------

        // -----------------------------------
        // 3. Generate vertical (k-neighbour-
        // hood) histogram.
        // -----------------------------------

        this._generate_sampleVarianceByK_histogram();
    }

    /**
     * Initializes horizontal histogram for sample variance per sample.
     * @private
     */
    _genenerate_sampleVarianceBySample_histogram()
    {
        // Create shorthand references.
        let dataset     = this._operator._dataset;
        let extrema     = dataset._cf_extrema;
        let dimensions  = dataset._cf_dimensions;
        // Use operator's target ID as group name.
        let dcGroupName = this._operator._target;

        // Generate dc.js chart object.
        this._sampleVarianceBySampleHistogram = dc.barChart("#" + this._divStructure.sampleHistogramDivID, dcGroupName);

        // Use arbitrary axis attribute for prototype purposes.
        let axesAttribute = "r_nx";

        // Create shorthand references.
        let key = axesAttribute + "#histogram";

        // Configure chart.
        this._sampleVarianceBySampleHistogram
            .height(40)
            // 0.8: Relative width of parent div.
            .width($("#" + this._divStructure.chartsContainerDivID).width())
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
     * @private
     */
    _generate_sampleVarianceByK_histogram()
    {
        // Create shorthand references.
        let dataset     = this._operator._dataset;
        let extrema     = dataset._cf_extrema;
        let dimensions  = dataset._cf_dimensions;
        // Use operator's target ID as group name.
        let dcGroupName = this._operator._target;

        // Generate dc.js chart object.
        this._sampleVarianceByKHistogram = dc.barChart("#" + this._divStructure.kHistogramDivID, dcGroupName);

        // Use arbitrary axis attribute for prototype purposes.
        let axesAttribute = "b_nx";

        // Create shorthand references.
        let key = axesAttribute + "#histogram";

        // Configure chart.
        this._sampleVarianceByKHistogram
            .height(40)
            .width($("#" + this._target).height())
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

    /**
     * Create (hardcoded) div structure for child nodes.
     * @returns {Object}
     */
    _createDivStructure()
    {
        let scope = this;

        // -----------------------------------
        // 1. Create charts container.
        // -----------------------------------

        let chartsContainerDiv  = Utils.spawnChildDiv(this._target, null, "dissonance-charts-container");
        let sampleHistogramDiv  = Utils.spawnChildDiv(chartsContainerDiv.id, null, "dissonance-variance horizontal");
        let heatmapDiv          = Utils.spawnChildDiv(chartsContainerDiv.id, null, "dissonance-heatmap");
        let kHistogramDiv       = Utils.spawnChildDiv(chartsContainerDiv.id, null, "dissonance-variance-chart vertical");

        // -----------------------------------
        // 2. Create title and options container.
        // -----------------------------------

        // Note: Listener for table icon is added by FilterReduceOperator, since it requires information about the table
        // panel.
        let infoDiv = Utils.spawnChildDiv(this._target, null, "dissonance-info");
        $("#" + infoDiv.id).html(
            "<span class='title'>" + scope._name + "</span>" +
            "<a id='dissonance-info-settings-icon' href='#'>" +
            "    <img src='./static/img/icon_settings.png' class='info-icon' alt='Settings' width='20px'>" +
            "</a>" +
            "<a id='dissonance-info-table-icon' href='#'>" +
            "    <img src='./static/img/icon_table.png' class='info-icon' alt='View in table' width='20px'>" +
            "</a>"
        )

        return {
            chartsContainerDivID: chartsContainerDiv.id,
            sampleHistogramDivID: sampleHistogramDiv.id,
            heatmapDivID: heatmapDiv.id,
            kHistogramDivID: kHistogramDiv.id
        };
    }

    render()
    {
        this._sampleVarianceBySampleHistogram.render();

        // Has to be drawn with updated height value.
        let newHeight = $("#" + this._target).height() - this._sampleVarianceBySampleHistogram.height();
        let newOffset = (
            newHeight / 4 +
            this._sampleVarianceBySampleHistogram.height() +
            this._sampleVarianceBySampleHistogram.margins().left -
            this._sampleVarianceBySampleHistogram.margins().right
        );
        this._sampleVarianceByKHistogram.width(newHeight);
        $("#" + this._divStructure.kHistogramDivID).css({
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
}