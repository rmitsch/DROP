import Scatterplot from "./Scatterplot.js";
import Utils from "../Utils.js";
import DRMetaDataset from "../data/DRMetaDataset.js";

/**
 * Scatterplots with dots connected by specified degree of freedom.
 */
export default class ParetoScatterplot extends Scatterplot
{
    /**
     * Instantiates new ParetoScatterplot.
     * @param name
     * @param panel
     * @param attributes Attributes to be considered in plot. Has to be of length 2. First argument is projected onto
     * x-axis, second to y-axis. Attributes can contain one hyperparameter and one objective or two objectives (might
     * produce unspecified behaviour if handled otherwise; currently not checked in code).
     * @param dataset
     * @param style Various style settings (chart width/height, colors, ...). Arbitrary format, has to be parsed indivdually
     * by concrete classes.
     * @param parentDivID
     * @param useBinning Defines whether (heaxagonal) binning should be used. If true, then points will be omitted.
     */
    constructor(name, panel, attributes, dataset, style, parentDivID, useBinning = false)
    {
        super(name, panel, attributes, dataset, style, parentDivID, useBinning);

        this._correlationBar              = null;
        this._correlationBarBackground    = null;
        this._initCorrelationBar();

        // If binning is required: Create separate, contained SVG.
        this._hexHeatmapContainerID = null;
        // Used to store max. value in heatmap cell - important for setting color of cells.
        this._maxCellValue = null;
        if (this._useBinning) {
            this._hexHeatmapContainerID = Utils.spawnChildDiv(this._target, null, 'pareto-scatterplot-hexheatmap').id;
        }

        // Update involved CSS classes.
        $("#" + this._target).addClass("pareto-scatterplot");
    }

    /**
     * Initializes divs for correlation bar.
     * @private
     */
    _initCorrelationBar()
    {
        // Spawn background for correlation bar and actual correlation bar.
        this._correlationBarBackground  = Utils.spawnChildDiv(this._target, null, "filter-reduce-correlation-bar");
        this._correlationBar            = Utils.spawnChildDiv(this._target, null, "filter-reduce-correlation-bar");
        $("#" + this._correlationBar.id).css("background-color", "#1f77b4", "height", "0");
    }

    /**
     * Fills correlation bar according to current correlation value for this chart's attributes.
     * @param correlations
     */
    updateCorrelationBar(correlations = null)
    {
        let attrX                   = this._axes_attributes.x.replace("*", "");
        let attrY                   = this._axes_attributes.y.replace("*", "");
        const attributeTranslation  = DRMetaDataset.translateAttributeNames(false);

        let correlationBar              = $("#" + this._correlationBar.id);
        let correlationBackgroundBar    = $("#" + this._correlationBarBackground.id);

        const correlation = correlations !== null ? correlations[attrX][attrY] : 0;

        const correlationText = "Correlation " +
            attributeTranslation[this._axes_attributes.x].toLowerCase() + " to " +
            attributeTranslation[this._axes_attributes.y].toLowerCase() + ": " +
            Math.round(correlation * 100) + "%";

        correlationBar.css("height", correlation * correlationBackgroundBar.height());
        correlationBar.attr("title", correlationText);
        correlationBackgroundBar.attr("title", correlationText);
    }

    render()
    {
        if (this._useBinning)
            $("#" + this._hexHeatmapContainerID).remove();

        this._cf_chart.render();

        if (this._useBinning) {
            this._hexHeatmapContainerID = Utils.spawnChildDiv(this._target, null, 'pareto-scatterplot-hexheatmap').id;
            this._drawHexagonalHeatmap();
            // Update z-index to allow selection in underlying map.
            $("#" + this._hexHeatmapContainerID).css("z-index", -1);
        }
    }

    constructCFChart()
    {
        // Use operator's target ID as group name.
        this._cf_chart = dc.scatterPlot(
            "#" + this._target,
            this._panel._operator._target,
            this._dataset,
            this._axes_attributes.x,
            this._axes_attributes.y,
            this._useBinning
        );

        // Create shorthand references.
        let instance    = this;
        let extrema     = this._dataset._cf_extrema;
        let intervals   = this._dataset._cf_intervals;
        let dimensions  = this._dataset._cf_dimensions;
        let key         = this._axes_attributes.x + ":" + this._axes_attributes.y;
        // Use padding so that first/last bar are not cut off in chart.
        let dataPadding = {
            x: intervals[this._axes_attributes.x] * this._style.paddingFactor,
            y: intervals[this._axes_attributes.y] * this._style.paddingFactor
        };

        // Configure chart.
        this._cf_chart
            .height(instance._style.height)
            // Take into account missing width in histograms.
            .width(instance._style.width - 8)
            .useCanvas(true)
            .x(d3.scale.linear().domain([
                extrema[instance._axes_attributes.x].min - dataPadding.x,
                extrema[instance._axes_attributes.x].max + dataPadding.x
            ]))
            .y(d3.scale.linear().domain([
                extrema[instance._axes_attributes.y].min - dataPadding.y,
                extrema[instance._axes_attributes.y].max + dataPadding.y
            ]))
            .xAxisLabel(instance._style.showAxisLabels.x ? instance._axes_attributes.x : null)
            .yAxisLabel(instance._style.showAxisLabels.y ? instance._axes_attributes.y : null)
            .renderHorizontalGridLines(true)
            .dimension(dimensions[key])
            .group(this._dataset.cf_groups[key])
            .existenceAccessor(function(d) {
                return d.value.items.length > 0;
            })
            .excludedSize(instance._style.excludedSymbolSize)
            .excludedOpacity(instance._style.excludedOpacity)
            .excludedColor(instance._style.excludedColor)
            .symbolSize(instance._style.symbolSize)
            .keyAccessor(function(d) {
                return d.key[0];
             })
            // Filter on end of brushing action, not meanwhile (performance drops massively otherwise).
            .filterOnBrushEnd(true)
            .mouseZoomable(false)
            .margins({ top: 0, right: 0, bottom: 16, left: 25 })
            .on('preRedraw', function(chart) {
                // If binning is used: Redraw heatmap.
                if (instance._useBinning) {
                    instance._drawHexagonalHeatmap();
                }
            })
            // Call cross-operator filter method on stage instance after filter event.
            .on("filtered", event => instance.propagateFilterChange(instance, key));

        // Hide axis lables, if so specified.
        if (!this._style.showAxisTickLabels.y)
            this._cf_chart.yAxis().tickFormat(function(v) { return ""; });
        if (!this._style.showAxisTickLabels.x)
            this._cf_chart.xAxis().tickFormat(function(v) { return ""; });

        // Set custom handler for filter events.
        if (!this._useBinning)
            this._setFilterHandler();

        // todo Mouseover for SVG how?
        this._cf_chart.selectAll('circle').on('mouseover', function() {
            d3.select(this).attr('fill', '#00c');
            console.log("mouseover")
        }).on('mouseout', function() {
            d3.select(this).attr('fill', 'orange')
        });

        // Set number of ticks for y-axis.
        this._cf_chart.yAxis().ticks(instance._style.numberOfTicks.y);
        this._cf_chart.xAxis().ticks(instance._style.numberOfTicks.x);

        // If this x-axis hosts categorical argument: Print categorical representations of numerical values.
        if (
            this._axes_attributes.x.indexOf("*") !== -1 &&
            instance._style.numberOfTicks.x &&
            this._style.showAxisTickLabels.x
        ) {
            // Get original name by removing suffix "*" from attribute name.
            let originalAttributeName = instance._axes_attributes.x.slice(0, -1);

            // Overwrite number of ticks with number of possible categorical values.
            this._cf_chart.xAxis().ticks(
                Object.keys(this._dataset.numericalToCategoricalValues[originalAttributeName]).length
            );

            // Use .tickFormat to convert numerical to original categorical representations.
            this._cf_chart.xAxis().tickFormat(function (tickValue) {
                // Print original categorical for this numerical representation.
                return tickValue in instance._dataset.numericalToCategoricalValues[originalAttributeName] ?
                        instance._dataset.numericalToCategoricalValues[originalAttributeName][tickValue] : "";
            });
        }
    }

    propagateFilterChange(instance, key)
    {
        const panel = this._panel;
        // Update correlation strengths, then re-render correlation bars.
        this._dataset.computeCorrelationStrengths(function (results) {
            panel.updateCorrelationBars(results);
        });

        return super.propagateFilterChange(instance, key);
    }

    /**
      * Updates data and buffered filter data after selection has been made in table panel.
      * @param embeddingIDs
      */
    updateFilteredRecordBuffer(embeddingIDs)
    {
        this._cf_chart.identifyFilteredRecords(d => embeddingIDs.has(d[2]));
    }

    /**
     * Configure's chart's filter handler so that series selection can be processed.
     * @private
     */
    _setFilterHandler()
    {
        let instance        = this;
        let stage           = this._panel._operator._stage;
        let chart           = this._cf_chart;
        let seriesMapping   = this.dataset._seriesMappingByHyperparameter[chart.variantAttribute];

        this._cf_chart.filterHandler(function (dimension, filters) {
            if (filters.length === 0) {
                chart.identifyFilteredRecords();
                dimension.filter(null);
                instance._panel.updateIDsToFilterInSSPs(null, instance._name);
            }

            else {
                let res = chart.identifyFilteredRecords(
                    function (d) {
                        for (let i = 0; i < filters.length; i++) {
                            let filter = filters[i];
                            if (filter.isFiltered && filter.isFiltered(d)) {
                                return true;
                            }
                            else if (filter <= d && filter >= d) {
                                return true;
                            }
                        }
                        return false;
                    }
                );

                // Gather set of all filtered IDs.
                let filteredIDs         = new Set();
                let filteredSeriesIDs   = new Set();

                for (const x in res) {
                    for (const y in res[x]) {
                        // https://stackoverflow.com/questions/32000865/simplest-way-to-merge-es6-maps-sets
                        filteredIDs = new Set(function*() {
                            yield* filteredIDs; yield* res[x][y].ids;
                        }());
                        filteredSeriesIDs = new Set(function*() {
                            yield* filteredSeriesIDs; yield* res[x][y].seriesIDs;
                        }());
                    }
                }

                // If modifying key down: Also select all records in same series as currently selected points.
                if (stage.shiftDown) {
                    let addedIDs = new Set();
                    for (let seriesID of filteredSeriesIDs) {
                        for (let id of seriesMapping.seriesToRecordMapping[seriesID]) {
                            filteredIDs.add(id);
                            addedIDs.add(id);
                        }
                    }

                    // Update chart's set of filtered points for rendering.
                    chart.addFilteredRecords(addedIDs);
                }

                // Update filtered IDs in other SSPs; update dimension's filter check.
                instance._panel.updateIDsToFilterInSSPs(filteredIDs, instance._name);
                dimension.filterFunction(d => filteredIDs.has(d[2]));
            }

            return filters;
        });
    }

    highlight(id, source)
    {
        if (source !== this._target) {
            this._cf_chart.highlight(id);
        }
    }

    /**
     * Draws hexagonal heatmap behind scatterplot. Uses existing chart SVG.
     * Source for heatmap code: https://bl.ocks.org/mbostock/4248145.
     * To refactor - move into HexagonalHeatmap.js.
     * @private
     */
    _drawHexagonalHeatmap()
    {
        // --------------------------------------
        // 1. Append/reset SVG to container div.
        // --------------------------------------

        let svg = d3.select("#" + this._hexHeatmapContainerID).select("svg");
        if (!svg.empty())
            svg.remove();
        // Append SVG.
        d3.select("#" + this._hexHeatmapContainerID).append("svg").attr("width", "100%").attr("height", "100%");
        svg = d3.select("#" + this._hexHeatmapContainerID).select("svg");

        // --------------------------------------
        // 2. Update size of container div and
        // SVG.
        // --------------------------------------

        // Container div.
        let heatmapContainer = $("#" + this._hexHeatmapContainerID);
        heatmapContainer.width(this._cf_chart.width() - this._cf_chart.margins().left - 1);
        heatmapContainer.height(this._cf_chart.height() - this._cf_chart.margins().bottom - 2);
        heatmapContainer.css("left", this._cf_chart.margins().left);
        // SVG.
        heatmapContainer.find("svg")[0].setAttribute('width', heatmapContainer.width());
        heatmapContainer.find("svg")[0].setAttribute('height', heatmapContainer.height());

        // --------------------------------------
        // 3. Bin filtered records.
        // --------------------------------------

        // Determine width - important for bin scaling.
        const margin  = {top: 0, right: 0, bottom: 0, left: 0};
        const width   = +svg.attr("width") - margin.left - margin.right;
        const height  = +svg.attr("height") - margin.top - margin.bottom;

        // Fetch all filtered records (crossfilter.all() doesn't consider filtering - why?).
        let instance    = this;
        const key       = this._axes_attributes.x + ":" + this._axes_attributes.y;
        let records     = this._dataset._cf_dimensions[key].top(Infinity);
        let dataPadding = {
            x: this._dataset._cf_intervals[this._axes_attributes.x] * this._style.paddingFactor,
            // Why does factor 1.5 work better for alignment than 1.0?
            y: this._dataset._cf_intervals[this._axes_attributes.y] * this._style.paddingFactor * 1.5
        };

        // Prepare data necessary for binning. Take padding values into account to ensure matching with histograms.
        let extrema = {
            x: {
                min: this._dataset._cf_extrema[this._axes_attributes.x].min - dataPadding.x,
                max: this._dataset._cf_extrema[this._axes_attributes.x].max + dataPadding.x,
            },
            y:  {
                min: this._dataset._cf_extrema[this._axes_attributes.y].min - dataPadding.y,
                max: this._dataset._cf_extrema[this._axes_attributes.y].max + dataPadding.y,
            }
        };

        // Calculate translations for binning so that value extrema match coordinate extrema.
        let translationFactors = {
            x: width / (this._dataset._cf_intervals[this._axes_attributes.x] + dataPadding.x * 2),
            y: height / (this._dataset._cf_intervals[this._axes_attributes.y] + dataPadding.y * 2),
        };

        let translateIntoCoordinates = function(d, axis) {
            return (
                d[instance._axes_attributes[axis]] - extrema[axis].min
            ) * translationFactors[axis];
        };

        // Loop through all records, calculate their coordinates.
        let recordCoords = [];
        for (let record of records) {
            recordCoords.push([
                translateIntoCoordinates(record, "x"),
                height - translateIntoCoordinates(record, "y")
            ]);
        }

        // Do actual binning.
        let hexbin = d3.hexbin()
            .radius(5)
            .extent([[0, 0], [width, height]]);
        let cells = hexbin(recordCoords);

        // Find max. value/number of values in cell, if not done yet. -> Do only once to initialize.
        if (this._maxCellValue === null) {
            this._maxCellValue = cells.reduce(
                (currIndex, maxCell, maxCellIndex, allCells) =>
                allCells[currIndex].length > maxCell.length ? currIndex : maxCellIndex, 0
            );
        }

        // --------------------------------------
        // 4. Draw heatmap.
        // --------------------------------------

        let g = svg.append("g").attr("transform", "translate(" + margin.left + "," + margin.top + ")");

        // Generate color scheme.
        let colors = d3
            .scaleLinear()
            .domain([0, Math.log10(this._dataset._cf_dimensions[key].top(Infinity).length)])
            .range(["#fff7fb", "#1f77b4"]);

        g.append("clipPath")
            .attr("id", "clip")
            .append("rect")
            .attr("width", width)
            .attr("height", height);

        g.append("g")
            .attr("class", "hexagon")
            .attr("clip-path", "url(#clip)")
            .selectAll("path")
            .data(cells)
            .enter().append("path")
            .attr("d", hexbin.hexagon())
            .attr("transform", d => "translate(" + d.x + "," + d.y + ")" )
            .attr("fill", d => colors(Math.log10(d.length)) );
    }

    resize(height = -1, width = -1)
    {
        if (height !== -1)
            this._cf_chart.height(height);
        if (width !== -1)
            this._cf_chart.width(width);

        // Note: prepareForResize() is necessary here since otherwise scale is updated only during render().
        // !this._useBinning only for performance reasons.
        if (!this._useBinning) {
            this._cf_chart.prepareForResize();
            this._cf_chart.updateFilteredRecordCoordinates();
        }

        this.render();
    }

    /**
     * Updates binning options in SSP chart.
     * @param config
     */
    updateSSPBinning(config)
    {
        // Ignore if settings are equal. Also ignore if .useBinning === true, i. e. this is a heatmap and not a SSP.
        if (
            (
                config.useLogs !== this._cf_chart.lineOptions.useLogs ||
                config.binFraction !== this._cf_chart.lineOptions.binFraction
            ) && !this._useBinning
        ) {
            this._cf_chart.lineOptions = config;
            this._cf_chart.render();
        }
    }
}