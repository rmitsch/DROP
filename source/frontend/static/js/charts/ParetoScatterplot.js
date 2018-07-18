import Scatterplot from "./Scatterplot.js";
import Utils from "../Utils.js";

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

        // If binning is required: Create separate, contained SVG.
        this._hexHeatmapContainerID = null;
        if (this._useBinning) {
            this._hexHeatmapContainerID = Utils.spawnChildDiv(this._target, null, 'pareto-scatterplot-hexheatmap').id;
        }

        // Update involved CSS classes.
        $("#" + this._target).addClass("pareto-scatterplot");
    }

    /**
     * Connects model parametrizations by specified attribute name. Used to e. g. draw pareto fronts.
     * @param hyperparameter Name of attribute to be used to connect models. Default value 'native' connects models with
     * diverging values for attribute on x-axis, but constant for all hyperparameter otherwise. Note that for
     * hyperparameter-objective plots all attributes other than native are ignored. Objective-objective plots accept all
     * available hyperparameters.
     */
    connectBy(hyperparameter = 'native')
    {
        throw new Error("ParetoScatterplot.connectBy(): Not implemented yet.");
    }

    render()
    {
        this._cf_chart.render();

        if (this._useBinning)
            this._drawHexagonalHeatmap(this._hexHeatmapContainerID);
    }

    constructCFChart()
    {
        // Use operator's target ID as group name.
        this._cf_chart = dc.scatterPlot(
            "#" + this._target,
            this._panel._operator._target,
            this._dataset,
            this._axes_attributes.x,
            this._useBinning
        );

        // Create shorthand references.
        let instance    = this;
        let extrema     = this._dataset._cf_extrema;
        let dimensions  = this._dataset._cf_dimensions;
        let key         = this._axes_attributes.x + ":" + this._axes_attributes.y;

        // Configure chart.
        this._cf_chart
            .height(instance._style.height)
            .width(instance._style.width)
            .useCanvas(true)
            .x(d3.scale.linear().domain(
                [extrema[instance._axes_attributes.x].min, extrema[instance._axes_attributes.x].max]
            ))
            .y(d3.scale.linear().domain(
                [extrema[instance._axes_attributes.y].min, extrema[instance._axes_attributes.y].max]
            ))
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
            // Filter on end of brushing action, not meanwhile (performance suffers otherwise).
            .filterOnBrushEnd(true)
            .mouseZoomable(false)
            .margins({top: 0, right: 0, bottom: 25, left: 25})
            .on('postRedraw', function(chart) {
                // If binning is used: Redraw heatmap.
                if (instance._useBinning) {

                }
            });

        // Set number of ticks for y-axis.
        this._cf_chart.yAxis().ticks(instance._style.numberOfTicks.y);
        this._cf_chart.xAxis().ticks(instance._style.numberOfTicks.x);

        // If this x-axis hosts categorical argument: Print categorical representations of numerical values.
        if (this._axes_attributes.x.indexOf("*") !== -1 && instance._style.numberOfTicks.x) {
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

    highlight(id, source)
    {
        if (source !== this._target) {
            this._cf_chart.highlight(id);
        }
    }

    /**
     * Draws hexagonal heatmap behind scatterplot. Uses existing chart SVG.
     * @param chartDivID
     * @private
     */
    _drawHexagonalHeatmap(chartDivID)
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
        heatmapContainer.width(this._cf_chart.width() - this._cf_chart.margins().left);
        heatmapContainer.height(this._cf_chart.height() - this._cf_chart.margins().bottom);
        heatmapContainer.css("left", this._cf_chart.margins().left);
        // SVG.
        heatmapContainer.find("svg")[0].setAttribute('width', heatmapContainer.width());
        heatmapContainer.find("svg")[0].setAttribute('height', heatmapContainer.height());


        // --------------------------------------
        // 3. Bin filtered records.
        // --------------------------------------

        // Determine width - important for bin scaling.
        let margin  = {top: 0, right: 0, bottom: 0, left: 0};
        let width   = +svg.attr("width") - margin.left - margin.right;
        let height  = +svg.attr("height") - margin.top - margin.bottom;

        // Fetch all filtered records (crossfilter.all() doesn't consider filtering - why?).
        let instance    = this;
        let key         = this._axes_attributes.x + ":" + this._axes_attributes.y;
        let records     = this._dataset._cf_dimensions[key].top(Infinity);

        // Prepare data necessary for binning.
        let extrema = {
            x: this._dataset._cf_extrema[this._axes_attributes.x],
            y: this._dataset._cf_extrema[this._axes_attributes.y],
        };
        // Calculate translation factors.
        let translationFactors = {
            x: width / (this._dataset._cf_intervals[this._axes_attributes.x] * 1.1),
            y: height / (this._dataset._cf_intervals[this._axes_attributes.y]),
        };
        // Calculate translations for binning so that value extrema match coordinate extrema.
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
                translateIntoCoordinates(record, "y")
            ]);
        }

        Next up: Draw recordCoords; then test and review next steps.

        // --------------------------------------
        // 4. Draw heatmap.
        // --------------------------------------

        let g = svg.append("g").attr("transform", "translate(" + margin.left + "," + margin.top + ")");

        var randomX = d3.randomNormal(width / 2, width / 2),
            randomY = d3.randomNormal(height / 2, height / 2),
            points = d3.range(2000).map(function() { return [randomX(), randomY()]; });

        var color = d3.scaleSequential(d3.interpolateLab("white", "steelblue"))
            .domain([0, 20]);

        var hexbin = d3.hexbin()
            .radius(6)
            .extent([[0, 0], [width, height]]);

        var x = d3.scaleLinear()
            .domain([0, width])
            .range([0, width]);

        var y = d3.scaleLinear()
            .domain([0, height])
            .range([height, 0]);

        g.append("clipPath")
            .attr("id", "clip")
            .append("rect")
            .attr("width", width)
            .attr("height", height);

        g.append("g")
            .attr("class", "hexagon")
            .attr("clip-path", "url(#clip)")
            .selectAll("path")
            .data(hexbin(points))
            .enter().append("path")
            .attr("d", hexbin.hexagon())
            .attr("transform", function(d) { return "translate(" + d.x + "," + d.y + ")"; })
            .attr("fill", function(d) { return color(d.length); });
    }
}