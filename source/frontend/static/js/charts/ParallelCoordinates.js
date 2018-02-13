import Chart from "./Chart.js";
import Utils from "../Utils.js";

/**
 * Parallel coordinates plot.
 * Utilizes https://github.com/syntagmatic/parallel-coordinates.
 */
export default class ParallelCoordinates extends Chart
{
    /**
     *
     * @param name
     * @param panel
     * @param attributes
     * @param dataset
     * @param style
     * @param parentDivID
     */
    constructor(name, panel, attributes, dataset, style, parentDivID)
    {
        super(name, panel, attributes, dataset, style, parentDivID);

        // Update involved CSS classes.
        $("#" + this._target).addClass("parcoords-container");

        // Check if attributes contain exactly one parameter.
        if (!Array.isArray(attributes) || attributes.length !== 2) {
            throw new Error("ParallelCoordinates: Has to be instantiated with an array of attributes with length 2.");
        }

        // Construct dictionary for axis/attribute names.
        this._axes_attributes = {
            x: attributes[0],
            y: attributes[1]
        };

        // Transform data so it's usable for parallel coordinate plot.
        this._dimensions        = {};
        this._transformedData   = this._transformData();
        // Store whether record IDs are filtered or not.
        this._filteredIDs       = new Set();
        this._updateFilteredIDs();

        // Construct chart.
        this.constructCFChart();

        // Implement methods necessary for dc.js hook and integrate it into it's chart registry.
        this._registerChartInDC();
    }

    /**
     * Transform data so it's usable for parallel coordinate plot.
     * @returns {Array}
     * @private
     */
    _transformData()
    {
        let seriesToRecordsMap = this._dataset._seriesMappingByHyperparameter[this._axes_attributes.x].seriesToRecordMapping;
        let transformedData = [];

        // --------------------------------------------------
        // 1. Transform records to format accepted by par-
        // coords chart.
        // --------------------------------------------------

        // Iterate over series.
        for (let i = 0; i < Object.values(seriesToRecordsMap).length; i++) {
            let transformedRecord = {ids: []};

            // Iterate over records in series.
            for (let j = 0; j < seriesToRecordsMap[i].length; j++) {
                let originalRecord = this._dataset.getDataByID(seriesToRecordsMap[i][j]);
                // Add corresponding record ID.
                transformedRecord.ids.push(originalRecord.id);

                // Since every dataset in loop has different value for categorical attribute:
                // Use values as attributes (i. e. we're pivoting the dataset while simultaneously discarding redundant
                // information.
                transformedRecord[originalRecord[this._axes_attributes.x]] = originalRecord[this._axes_attributes.y];
            }

            transformedData.push(transformedRecord);
        }

        // --------------------------------------------------
        // 2. Create dimensions.
        // --------------------------------------------------

        // Create dimensions. Sort alphabetically to keep sequence of dimensions consistent with dc.js' charts' axes.
        let xAttributeValuesSorted = Object.keys(transformedData[0]);
        xAttributeValuesSorted.sort();
        let attributeIndex = 0;
        // Create dimensions, sorted lexically by value of attribute on x-axis.
        for (let index in xAttributeValuesSorted) {
            let key = xAttributeValuesSorted[index];
            if (key !== "ids") {
                this._dimensions[key] = {
                    index: attributeIndex++,
                    title: key,
                    orient: "left",
                    type: "number",
                    ticks: 0,
                    extrema: {
                        min: Number.MAX_VALUE,
                        max: -Number.MAX_VALUE
                    }
                };
            }
        }

        // --------------------------------------------------
        // 3. Determine extrema. Add values to dimensions
        // objects.
        // --------------------------------------------------

        for (let i = 0; i < transformedData.length; i++) {
            for (let dimKey in this._dimensions) {
                let value   = transformedData[i][dimKey];
                let currMin = this._dimensions[dimKey].extrema.min;
                let currMax = this._dimensions[dimKey].extrema.max;

                this._dimensions[dimKey].extrema.min = value < currMin ? value : currMin;
                this._dimensions[dimKey].extrema.max = value > currMax ? value : currMax;
            }
        }

        return transformedData;
    }

    render()
    {
        this._cf_chart.render();
    }

    constructCFChart()
    {
        let instance = this;

        // Construct conatiner div for parcoords element.
        let div = Utils.spawnChildDiv(this._target, null, 'parcoords');
        // Get dimension holding both dimensions used for parallel coordinates chart.
        let dim = instance._dataset._cf_dimensions[instance._axes_attributes.x + ":" + instance._axes_attributes.y];

        // Use config object to pass information useful at initialization time.
        this._cf_chart = d3.parcoords({
                dimensions: this._dimensions,
                data: this._transformedData,
                colorRange: ["blue#ccc", "#cccblue", "#ccc#ccc", "blueblue"]
            })("#" + div.id)
            .height(this._style.height)
            .width(this._style.width)
            .hideAxis(["ids"])
            .alpha(0.015)
            .composite("darken")
            .color(function(d) { return "blue"; })
            // Define colors for ends.
            .colors(function(d) {
                // Assign blue for filtered and grey for unfiltered records
                return d.ids.map(id => instance._filteredIDs.has(id) ? "blue" : "#ccc");
            })
            .margin({top: 5, right: 0, bottom: 18, left: 0})
            .mode("queue")
            .on("render", Utils.debounce(function() {
                // Update pareto frontiers after rendering.
                let filteredData = instance._sortCrossfilterDataByObjective();
                instance._cf_chart.renderParetoFrontiers(
                    "brushed",
                    filteredData[0][instance._axes_attributes.y],
                    filteredData[filteredData.length - 1][instance._axes_attributes.y]);
                // USE SCATTERPLOTS WITH EQUIDISTANT X-COORDINATES for categorical variables!
                // NEXT UP: aggregate dictionary with min/max values for transformed data values (cf.group with condtions?)
            }, 0))
            .on("brush", Utils.debounce(function(data) {
                // Get brushed thresholds for involved dimensions (e. g. objectives).
                let brushedThresholds = instance._cf_chart.brushExtents();
                // Filter dimension for this objective by the selected objective thresholds.
                // Return true if datapoint's value on y-axis lies in interval defined by user on the corrsponding
                // x-axis.
                dim.filter(function(d) {
                    // d[0] corresponds to value on x-axis, d[1] to value on y-axis.

                    // If value on x-axis not selected by user: Filter all.
                    if (!(d[0] in brushedThresholds))
                        return true;

                    // Otherwise: Check if value is inside interval.
                    return d[1] >= brushedThresholds[d[0]][0] && d[1] <= brushedThresholds[d[0]][1];
                });

                // Redraw all charts after filter operation.
                dc.redrawAll(instance._panel._operator._target);

                // Update pareto frontiers after rendering.
                let filteredData = instance._sortCrossfilterDataByObjective();
                instance._cf_chart.renderParetoFrontiers(
                    "brushed",
                    filteredData[0][instance._axes_attributes.y],
                    filteredData[filteredData.length - 1][instance._axes_attributes.y]
                );
            }, 250))
            .brushMode("1D-axes");

        // this._cf_chart.svg.selectAll("text").style("font", "10px sans-serif");
    }

    /**
     * Implement methods necessary for dc.js hook and integrate it into it's chart registry.
     */
    _registerChartInDC()
    {
        // --------------------------------
        // 1. Implement necessary elements
        // of dc.js' interface for charts.
        // --------------------------------

        let instance = this;

        this._cf_chart.redraw       = function() {
            // Update filtered IDs.
            instance._updateFilteredIDs();

            // Redraw chart.
            instance._cf_chart.render();
        };

        this._cf_chart.filterAll    = function() {
            // Set all records as filtered.
            instance._filteredIDs = new Set(instance._dataset._data.map(record => +record.id))
            // Reset brush.
            instance._cf_chart.brushReset();
        };

        // --------------------------------
        // 2. Register parcoords plot in
        // dc.js' registry.
        // --------------------------------

        // Use operators ID as group ID (all panels in operator use the same dataset and therefore should be notified if
        // filter conditions change).
        dc.chartRegistry.register(this._cf_chart, this._panel._operator._target);
    }

    /**
     * Updates filtered IDs.
     * @private
     */
    _updateFilteredIDs()
    {
        // Get filtered items, fill dictionary for filtered records' IDs.
        let filteredItems = this._dataset._cf_dimensions[this._axes_attributes.x].top(Infinity);

        // Reset dictionary with filtered IDs.
        this._filteredIDs = new Set(filteredItems.map(record => +record.id));
    }

    /**
     * Sorts data in crossfilter by this chart's objective.
     * @private
     */
    _sortCrossfilterDataByObjective()
    {
        // Update pareto frontiers after rendering.
        let filteredData = this._dataset._cf_dimensions[this._axes_attributes.x].top(Infinity);

        // Sort data by number of entries in this attribute's histogram.
        let instance = this;
        filteredData.sort(function(entryA, entryB) {
            let objectiveA = entryA[instance._axes_attributes.y];
            let objectiveB = entryB[instance._axes_attributes.y];

            return objectiveA > objectiveB ? 1 : (objectiveB > objectiveA ? -1 : 0);
        });

        return filteredData;
    }
}