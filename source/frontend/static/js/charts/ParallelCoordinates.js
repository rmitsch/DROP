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
                    ticks: 0
                };
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

        // Use config object to pass information useful at initialization time.
        this._cf_chart = d3.parcoords({
                dimensions: this._dimensions,
                data: this._transformedData,
                colorRange: ["blue#ccc", "#cccblue", "#ccc#ccc", "blueblue"]
            })("#" + div.id)
            .height(this._style.height)
            .width(this._style.width)
            .hideAxis(["ids"])
            .alpha(0.04)
            .composite("darken")
            .color(function(d) { return "blue"; })
            // Define colors for ends.
            .colors(function(d) {
                // Assign blue for filtered and grey for unfiltered records
                return d.ids.map(id => instance._filteredIDs.has(id) ? "blue" : "#ccc");
            })
            .margin({top: 5, right: 0, bottom: 18, left: 0})
            .mode("queue")
            .on("brush", Utils.debounce(function(data) {
                // Get brushed thresholds for involved dimensions (e. g. objectives).
                let brushedThresholds = instance._cf_chart.brushExtents();
                // Get dimension holding both dimensions used for parallel coordinates chart.
                let dim = instance._dataset._cf_dimensions[instance._axes_attributes.x + ":" + instance._axes_attributes.y];
                // Aggregate conditions.
                let conditions = {};
                for (let xValue in brushedThresholds) {
                    if (brushedThresholds.hasOwnProperty(xValue)) {
                        conditions[xValue] = brushedThresholds[xValue];
                    }
                }
                console.log(brushedThresholds);

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
}