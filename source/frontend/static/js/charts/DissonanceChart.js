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

        // Define color range and related values.
        this._colors                = d3.scaleLinear().range(["#fff7fb", "#1f77b4"]);
        this._highlightColors       = d3.scaleLinear().range(["#fee5d9", "#ff0000"]);
        this._numRecordsInEmbedding = this._dataset._data.length / this._dataset._drModelMetadata._data.length;

        // Cache number of filtered records (to speed up color computation).
        this._numFilteredRecords = this._dataset._cf_dimensions.model_id.top(Infinity).length;

        // Constant width in pixel heatmap SVG is too large.
        this._heatmapCutoff = 20;

        // Generate div structure for child nodes.
        this._divStructure = this._createDivStructure();

        // Track whether charts have been constructed yet.
        this._areChartsConstructed = false;

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
        const numCols = this._dataset._binCounts.x;
        const numRows = this._dataset._binCounts.y;

        // Update cached number of filtered records.
        this._numFilteredRecords = this._dataset._cf_dimensions.model_id.top(Infinity).length;

        // Use heatmap width and height as yard stick for histograms.
        let newHeight       = Math.floor(
            ($("#" + this._panel._target).height() - 90) / numRows
        ) * numRows;
        let newWidth        = Math.floor(
            ($("#" + this._target).width()) / numCols
        ) * numCols - 140;

        // -------------------------------
        // 1. Render horizontal histogram.
        // -------------------------------

        this._horizontalHistogram.width(
            newWidth +
            this._horizontalHistogram.margins().left +
            this._horizontalHistogram.margins().right -
            this._heatmapCutoff + 2
        );
        !this._areChartsConstructed ? this._horizontalHistogram.render() : this._horizontalHistogram.redraw();

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
                this._heatmapCutoff + 3
            ) + "px"
        });
        !this._areChartsConstructed ? this._verticalHistogram.render() : this._verticalHistogram.redraw();

        // -------------------------------
        // 3. Render heatmap and color
        //    scale.
        // -------------------------------

        this._dissonanceHeatmap.width(newWidth);
        this._dissonanceHeatmap.height(newHeight);
        this.renderHeatmap();

        // Adjust color scale height.
        $("#" + this._divStructure.colorPaletteDiv.id).height(newHeight);
        // Adjust color scale's labels' positions.
        for (let label of $(".color-palette-label")) {
            let labelElement = $("#" + label.id);
            labelElement.css("top", labelElement.parent().height() / 2 - labelElement.height() / 2);
        }

        this._areChartsConstructed = true;
    }

    renderHeatmap()
    {
        !this._areChartsConstructed ? this._dissonanceHeatmap.render() : this._dissonanceHeatmap.redraw();
    }

    /**
     * Compute colors for heatmap cell. Uses logarithmic scaling.
     * @param d
     * @private
     * @returns string Color as hex value.
     */
    _computeColorsForCell(d)
    {
        const colorDomain = this._colors.domain(
            [0, Math.log10(this._numFilteredRecords)]
        );

        return d === 0 ? "#fff" : colorDomain(Math.log10(d));
    }

    /**
     * Compute colors for heatmap cell when highlighted. Uses logarithmic scaling.
     * @param d
     * @private
     * @returns string Color as hex value.
     */
    _computeHighlightColorsForCell(d)
    {
        const colorDomain = this._highlightColors.domain(
            [0, Math.log10(this._numRecordsInEmbedding)]
        );

        return d === 0 ? "#fff" : colorDomain(Math.log10(d));
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
        let scope       = this;
        let dataset     = this._dataset;
        let dimensions  = dataset._cf_dimensions;
        let attribute   = "samplesInModelsMeasure:sampleDRModelMeasure";

        // Configure chart.
        this._dissonanceHeatmap
            .height(300)
            .width(300)
            .dimension(dimensions[attribute + "#sort"])
            .group(Utils.fillDissonanceHeatmapGroup(
                dataset._cf_groups[attribute],
                {min: 0, max: dataset._binCounts.x},
                {min: 0, max: dataset._binCounts.y}
            ))
            // Not quite clear why d.value sometimes doesn't have .count attribute with custom reduce function.
            // Using workaround here.
            .colorAccessor(d => typeof d.value === 'number' ? d.value : d.value.count)
            .colors(d => scope._computeColorsForCell(d))
            .keyAccessor(function(d)    { return d.key[0];  })
            .valueAccessor(function(d)  { return d.key[1]; })
            .title(function(d)          { return ""; })
            // Supress column/row label output.
            .colsLabel(function(d)      { return ""; })
            .rowsLabel(function(d)      { return ""; })
            .margins({top: 0, right: 20, bottom: 0, left: 0})
            .transitionDuration(0)
            // Cut off superfluous SVG height (probably reserved for labels).
            // Note: Has to be tested with different widths and heights.
            .on('postRedraw', function(chart) {
                let svg = $("#" + scope._divStructure.heatmapDivID).find('svg')[0];
                svg.setAttribute('width', (svg.width.baseVal.value - scope._heatmapCutoff) + "px");
            })
            .on('postRender', function(chart) {
                let svg = $("#" + scope._divStructure.heatmapDivID).find('svg')[0];
                svg.setAttribute('width', (svg.width.baseVal.value - scope._heatmapCutoff) + "px");
            });

        // Forward cell selection to filter mechanism in DissonanceDataset.
        this._dissonanceHeatmap.boxOnClick(function (d) {
            console.log(d);
            dc.events.trigger(function () {
                // dataset.addToHeatmapCellSelection(d.key)
                // scope._dissonanceHeatmap.filter(d.key);
                // scope._dissonanceHeatmap.redrawGroup();
            });
        });

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

        // Generate dc.js chart object.
        this._horizontalHistogram = dc.barChart(
            "#" + this._divStructure.horizontalHistogramDivID,
            dcGroupName
        );

        // Configure chart.
        this._horizontalHistogram
            .height(40)
            .width(Math.floor($("#" + this._target).width() / dataset._binCounts.x) * dataset._binCounts.x)
            .keyAccessor( function(d) { return d.key; } )
            .valueAccessor( d => d.value.count)
            .elasticY(false)
            .x(d3.scale.linear().domain([0, dataset._binCounts.x]))
            .y(d3.scale.linear().domain([0, extrema[yAttribute].max]))
            .brushOn(true)
            .filterOnBrushEnd(true)
            .dimension(dimensions[xAttribute + "#sort"])
            .group(dataset._cf_groups[yAttribute])
            .margins({top: 5, right: 5, bottom: 5, left: 40})
            .brushOn(false)
            .gap(0);

        // Set bar width.
        this._horizontalHistogram.xUnits(dc.units.fp.precision(1));
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

        // Generate dc.js chart object.
        this._verticalHistogram = dc.barChart(
            "#" + this._divStructure.verticalHistogramDivID,
            dcGroupName
        );

        // Configure chart.
        this._verticalHistogram
            .height(40)
            .width($("#" + this._panel._target).height())
            .valueAccessor(d => d.value.count)
            .elasticY(false)
            .x(d3.scale.linear().domain([0, dataset._binCounts.y]))
            .y(d3.scale.linear().domain([0, extrema[yAttribute].max]))
            .brushOn(true)
            .filterOnBrushEnd(true)
            .dimension(dimensions[xAttribute + "#sort"])
            .group(dataset._cf_groups[yAttribute])
            .brushOn(false)
            .margins({top: 5, right: 5, bottom: 5, left: 35})
            .gap(0);

        // Set bar width.
        // this._verticalHistogram.xUnits(dc.units.fp.precision(binWidth));
        this._verticalHistogram.xUnits(dc.units.fp.precision(1));
        // Set tick format on y-axis.
        this._verticalHistogram.yAxis().tickFormat(d3.format('.3s'));
        // Set number of ticks.
        this._verticalHistogram.yAxis().ticks(1);
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
        let paletteDiv          = Utils.spawnChildDiv(this._target, null, "color-palette");

        const numRecords    = this._dataset._cf_dimensions.model_id.top(Infinity).length;
        const numSteps      = Math.floor(Math.log10(numRecords));

        let colorToPaletteCellMap = {};
        for (let i = numSteps - 1; i >= 0; i--) {
            let color                       = this._computeColorsForCell(numRecords / Math.pow(10, numSteps - i));
            let cell                        = Utils.spawnChildDiv(paletteDiv.id, null, "color-palette-cell");
            colorToPaletteCellMap[color]    = cell.id;

            // Set color and height of cell.
            const cellDiv = $("#" + cell.id);
            cellDiv.css("background-color", color);
            cellDiv.css("height", "20%");

            // Create labels indicating color <-> percentage mapping.
            if (i === numSteps - 1 ||
                i === Math.floor(numSteps / 2) ||
                i === 0
            ) {
                Utils.spawnChildSpan(cell.id, null, "color-palette-label", Math.pow(10, -(numSteps - i) + 2) + "%");
            }
        }

        // Create axis labels.
        let xAxisLabel = Utils.spawnChildDiv(
            this._target, null, "dissonance-chart-x-axis-label", "Metric per record in embedding"
        );
        let yAxisLabel = Utils.spawnChildDiv(
            this._target, null, "dissonance-chart-y-axis-label", "Embedding quality"
        );

        return {
            horizontalHistogramDivID: sampleHistogramDiv.id,
            heatmapDivID: heatmapDiv.id,
            verticalHistogramDivID: kHistogramDiv.id,
            colorPaletteDiv: {
                id: paletteDiv.id,
                cells: colorToPaletteCellMap,
                labels: null
            }
        };
    }

    /**
     * Orders all charts by specified sorting criterion.
     * @param orderCriterion Possible values:
     *  - "sim-quality" for sample-in-model quality (horizontal barchart),
     *  - "m-quality" for model quality (vertical barchart),
     *  - "cluster" for sorting by strongest clusters in heatmap,
     *  - "natural" for natural sorting (i. e. by values instead counts of values).
     */
    orderBy(orderCriterion)
    {
        let dataset = this._dataset;

        // -----------------------------------------------------------
        // 1. Check if sort settings are valid and have changed.
        //    If not, abort/exit.
        // ----------------------------------------------------------

        // Check for validity of specified sorting criterion.
        if (!dataset._allowedCriterions.includes(orderCriterion.x) ||
            !dataset._allowedCriterions.includes(orderCriterion.y)
        )
            throw new RangeError("Invalid value for DissonanceChart's sort criterion chosen.");

        // Check if settings have changed.
        if (dataset._sortSettings.horizontal.criterion === orderCriterion.x &&
            dataset._sortSettings.vertical.criterion === orderCriterion.y
        )
            return;

        // -----------------------------------------------------------
        // 2. If settings have changed: Update sorting and re-render.
        // ----------------------------------------------------------

        // Update sorting of horizontal histogram.
        if (dataset._sortSettings.horizontal.criterion !== orderCriterion.x) {
            // Set sort order for histogram.
            this._horizontalHistogram.group(dataset.sortHistogramGroup(
                dataset._cf_groups["samplesInModels#measure"],
                dataset._sortSettings.horizontal,
                orderCriterion.x,
                {min: 0, max: this._dataset._binCounts.x}
            ));
            this._horizontalHistogram.render();
            this._verticalHistogram.render();
        }

        // Update sorting of verticalhistogram.
        if (dataset._sortSettings.vertical.criterion !== orderCriterion.y) {
            this._verticalHistogram.group(dataset.sortHistogramGroup(
                dataset._cf_groups["samplesInModels#" + this._dataset._supportedDRModelMeasure],
                dataset._sortSettings.vertical,
                orderCriterion.y,
                {min: 0, max: this._dataset._binCounts.y}
            ));
            this._verticalHistogram.render();
            this._horizontalHistogram.render();
        }

        // In all cases: heatmap has to be re-ordered.
        this._dissonanceHeatmap.group(
            Utils.fillDissonanceHeatmapGroup(
                dataset.sortHeatmapGroup(
                    dataset._cf_groups["samplesInModelsMeasure:sampleDRModelMeasure"],
                    dataset._sortSettings.heatmap
                ),
                {min: 0, max: dataset._binCounts.x},
                {min: 0, max: dataset._binCounts.y}
            )
        );
        this._dissonanceHeatmap.render();
    }

    resize()
    {
        let panelDiv = $("#" + this._target);
        console.log(panelDiv.height(), panelDiv.width(), this._lastPanelSize)
        if (panelDiv.height() !== this._lastPanelSize.height ||
            panelDiv.width() !== this._lastPanelSize.width) {
            this.render();
        }

        // Store size of panel at time of last render.
        this._lastPanelSize.width   = panelDiv.width();
        this._lastPanelSize.height  = panelDiv.height();
    }

    highlight(id, source)
    {
        let scope = this;

        if (source !== this._name) {
            if (id !== null) {
                this._verticalHistogram.selectAll('rect.bar').each(function(d) {
                    if (id in d.data.value.ids)
                        d3.select(this).attr("fill", "red");
                });
                this._horizontalHistogram.selectAll('rect.bar').each(function(d) {
                    if (id in d.data.value.ids)
                        d3.select(this).attr("fill", scope._computeHighlightColorsForCell(d.data.value.ids[id]));
                });
                this._dissonanceHeatmap.selectAll('rect.heat-box').each(function(d) {
                    if (typeof d.value === "object" && id in d.value.ids)
                        d3.select(this).attr("fill", scope._computeHighlightColorsForCell(d.value.ids[id]));
                });
            }

            // Reset all bars to default colors.
            else {
                this._verticalHistogram.selectAll('rect.bar').each(function(d) {
                    d3.select(this).attr("fill", "#1f77b4");
                });
                this._horizontalHistogram.selectAll('rect.bar').each(function(d) {
                    d3.select(this).attr("fill", "#1f77b4");
                });
                this._dissonanceHeatmap.selectAll('rect.heat-box').each(function(d) {
                    d3.select(this).attr(
                        "fill", scope._computeColorsForCell(typeof d.value === 'number' ? d.value : d.value.count)
                    );
                });
            }
        }
    }
}