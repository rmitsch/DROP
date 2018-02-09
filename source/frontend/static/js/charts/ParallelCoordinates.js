import Chart from "./Chart.js"
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

        // Construct chart.
        this.constructCFChart();
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

        // Create dimensions
        for (let key in transformedData[0]) {
            if (key !== "ids") {
                this._dimensions[key] = {
                    title: " ",
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
        // Construct conatiner div for parcoords element.
        let div = Utils.spawnChildDiv(this._target, null, 'parcoords');

        this._cf_chart = d3.parcoords()("#" + div.id);
        this._cf_chart
            .data(this._transformedData)
            .height(this._style.height)
            .width(this._style.width)
            .hideAxis(["ids"])
            .alpha(0.045)
            .composite("darken")
            .dimensions(this._dimensions)
            .margin({top: 5, right: 0, bottom: 18, left: 0})
            .render()
            .mode("queue")
            .brushMode("1D-axes");

        // this._cf_chart.svg.selectAll("text").style("font", "10px sans-serif");
    }
}