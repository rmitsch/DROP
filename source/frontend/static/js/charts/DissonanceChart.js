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

        // Constant width in pixel heatmap SVG is too wide.
        this._heatmapCutoff = 20;
        // Constant for color scheme.
        this._colorScheme = ["#fff", "#fff7fb","#ece7f2","#d0d1e6","#a6bddb","#74a9cf","#3690c0","#0570b0","#045a8d","#023858"];
        // Define color domain.
        this._colorDomain = this._calculateColorDomain();

        // Generate div structure for child nodes.
        this._divStructure = this._createDivStructure();

        // Construct graph.
        this.constructCFChart();
    }

    /**
     * Calculates color domain based on existing color scheme and data extrema.
     * @private
     */
    _calculateColorDomain()
    {
        let colorDomain = [0];
        let extrema     = this._dataset._cf_extrema["samplesInModelsMeasure:sampleDRModelMeasure"];
        for (let i = 1; i < this._colorScheme.length; i++) {
            colorDomain.push(extrema.max / (this._colorScheme.length - 1) * i);
        }

        return colorDomain;
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
        let numCols         = this._dataset._binCounts.x;
        let numRows         = this._dataset._binCounts.y;
        // Use heatmap width and height as yard stick for histograms.
        let newHeight       = Math.floor(
            ($("#" + this._panel._target).height() * 0.9 - 55) / numRows
        ) * numRows;
        let newWidth        = Math.floor(
            $("#" + this._target).width() * 0.95 / numCols
        ) * numCols;

        // -------------------------------
        // 1. Render horizontal histogram.
        // -------------------------------

        this._horizontalHistogram.width(
            newWidth +
            this._horizontalHistogram.margins().left +
            this._horizontalHistogram.margins().right -
            this._heatmapCutoff + 2
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
                this._heatmapCutoff + 3
            ) + "px"
        });
        this._verticalHistogram.render();

        // -------------------------------
        // 3. Render heatmap and color
        //    scale.
        // -------------------------------

        this._dissonanceHeatmap.width(newWidth);
        this._dissonanceHeatmap.height(newHeight);
        this._dissonanceHeatmap.render();

        // Adjust color scale height.
        $("#" + this._divStructure.colorPaletteDiv.id).height(newHeight);
        // Adjust color scale's labels' positions.
        for (let label of $(".color-palette-label")) {
            let labelElement = $("#" + label.id);
            labelElement.css("top", labelElement.parent().height() / 2 - labelElement.height() / 2);
        }
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
                    .domain(this._colorDomain)
                    .range(this._colorScheme)
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
            .keyAccessor( function(d) { return d.key; } )
            .valueAccessor( function(d) { return d.value; } )
            .elasticY(false)
            .x(d3.scale.linear().domain([0, dataset._binCounts.x])) // extrema[xAttribute].max]
            .y(d3.scale.linear().domain([0, extrema[yAttribute].max]))
            .brushOn(true)
            .filterOnBrushEnd(true)
            .dimension(dataset._cf_dimensions[xAttribute + "#sort"]) // dimensions[xAttribute]
            .group(dataset.sortGroup(dataset._cf_groups[yAttribute], "asc"))
            .margins({top: 5, right: 5, bottom: 5, left: 40})
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

        // Generate divs inside palette - one for each color.
        let colorToPaletteCellMap = {};
        for (let i = this._colorScheme.length - 1; i >= 0; i--) {
            let color                       = this._colorScheme[i];
            let cell                        = Utils.spawnChildDiv(paletteDiv.id, null, "color-palette-cell");
            colorToPaletteCellMap[color]    = cell.id;

            // Set color of cell.
            $("#" + cell.id).css("background-color", color);

            // Create labels indicating color <-> percentage mapping.
            if (i === this._colorScheme.length - 1 ||
                i === Math.round(this._colorScheme.length / 2) ||
                i === 0
            ) {
                let percentage = (this._colorDomain[i] / this._dataset._crossfilter.all().length) * 100;
                // Spawn label.
                Utils.spawnChildSpan(cell.id, null, "color-palette-label", Math.round(percentage) + "%");
            }
        }

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
        switch (orderCriterion) {
            case "natural":
                console.log("natural sort")
                this._horizontalHistogram.ordering(dc.pluck('value'));
                this._horizontalHistogram.renderGroup();
                break;

            case "sim-quality":
                console.log("sim sort")


                break;

            case "m-quality":
                break;

            case "cluster":
                break;

            default:
                throw new RangeError("Invalid value for DissonanceChart's sort criterion chosen.");
        }
    }
}