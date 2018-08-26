import Panel from "./Panel.js";
import Utils from "../Utils.js";
import DRMetaDataset from "../data/DRMetaDataset.js";

/**
 * Panel for model detail view.
 */
export default class ModelDetailPanel extends Panel
{
    /**
     * Constructs new panel for model detail view charts.
     * @param name
     * @param operator
     * @param parentDivID
     */
    constructor(name, operator, parentDivID)
    {
        super(name, operator, parentDivID);

        // Update involved CSS classes.
        $("#" + this._target).addClass("model-detail-panel");

        // Create div structure for child nodes.
        this._divStructure = this._createDivStructure();
    }

    /**
     * Generates all chart objects. Does _not_ render them.
     */
    _generateCharts()
    {
        console.log("Generating ModelDetailPanel...");
    }

    /**
     * Create (hardcoded) div structure for child nodes.
     * @returns {Object}
     */
    _createDivStructure()
    {
        // -----------------------------------
        // 1. Create panes.
        // -----------------------------------

        // Left pane.
        let parameterPane = Utils.spawnChildDiv(this._target, "model-detail-parameter-pane", "split split-horizontal");
        // Right pane.
        let samplePane = Utils.spawnChildDiv(this._target, "model-detail-sample-pane", "split split-horizontal");

        // 1. Upper-left pane - hyperparameters and objectives for current DR model.
        let attributePane = Utils.spawnChildDiv(
            parameterPane.id, null, "model-detail-pane split split-vertical",
            `<div class='model-details-block reduced-padding'>
                <div class='model-details-title'>Hyperparameters</div>
                <div id="model-details-block-hyperparameter-content"></div>
                <hr>
                <div class='model-details-title'>Objectives</div>
                <div id="model-details-block-objective-content"></div>
            </div>`
        );
        // 2. Bottom-left pane - explanation of hyperparameter importance for this DR model utilizing LIME.
        let limePane = Utils.spawnChildDiv(
            parameterPane.id, null, "model-detail-pane split-vertical",
            `<div class='model-details-block'>
                <div class='model-details-title'>Local Hyperparameter Relevance</div>
            </div>`
        );

        // 3. Upper-right pane - all records in scatterplot (SPLOM? -> What to do with higher-dim. projections?).
        let scatterplotPane = Utils.spawnChildDiv(
            samplePane.id, null, "model-detail-pane split-vertical",
            `<div class='model-details-block reduced-padding'>
                <div class='model-details-title'>All Records</div>
            </div>`
        );

        // Bottom-right pane - detailed information to currently selected record.
        let recordPane = Utils.spawnChildDiv(
            samplePane.id, null, "model-detail-pane split-vertical",
            `<div class='model-details-block'>
                <div class='model-details-title'>Selected Sample(s)</span>
            </div>`
        );

        // -----------------------------------
        // 2. Configure splitting.
        // -----------------------------------

        // Split left and right pane.
        Split(["#" + parameterPane.id, "#" + samplePane.id], {
            direction: "horizontal",
            sizes: [25, 75],
            onDragEnd: function() {}
        });

        // Split upper-left and bottom-left pane.
        Split(["#" + attributePane.id, "#" + limePane.id], {
            direction: "vertical",
            sizes: [50, 50],
            onDragEnd: function() {}
        });

        // Split upper-right and bottom-right pane.
        Split(["#" + scatterplotPane.id, "#" + recordPane.id], {
            direction: "vertical",
            sizes: [50, 50],
            onDragEnd: function() {}
        });

        // Return all panes' IDs.
        return {
            parameterPaneID: parameterPane.id,
            samplePaneID: samplePane.id,
            attributePane: {
                id: attributePane.id,
                hyperparameterContentID: "model-details-block-hyperparameter-content",
                objectiveContentID: "model-details-block-objective-content",
            },
            limePaneID: limePane.id,
            scatterplotPaneID: scatterplotPane.id,
            recordPaneID: recordPane.id
        };
    }

    render()
    {
        // -------------------------------------------------------
        // 1. Draw sparklines for attributes.
        // -------------------------------------------------------

        this._drawAttributeSparklines();

        // -------------------------------------------------------
        // 2. Draw scatterplot/SPLOM showing individual records.
        // -------------------------------------------------------

        this._drawRecordScatterplots();

        // -------------------------------------------------------
        // 3. Fill table with individual, selected records.
        // -------------------------------------------------------

        this._fillRecordTable();
    }

    _fillRecordTable()
    {
        let scope               = this;
        let drMetaDataset       = this._operator._drMetaDataset;
        let origDataset         = this._data._originalDataset;
        // Fetch metadata structure (i. e. attribute names and types).
        let metadataStructure   = drMetaDataset._metadata;
        // Fetch divs containing attribute sparklines.
        let chartContainerDiv   = $("#" + this._divStructure.scatterplotPaneID);
    }

    _drawRecordScatterplots()
    {
        let scope               = this;
        let drMetaDataset       = this._operator._drMetaDataset;
        // Fetch metadata structure (i. e. attribute names and types).
        let metadataStructure   = drMetaDataset._metadata;
        // Fetch divs containing attribute sparklines.
        let chartContainerDiv   = $("#" + this._divStructure.scatterplotPaneID);

        // -------------------------------------------------------
        // 1. Reset existing chart container.
        // -------------------------------------------------------

        // Reset chart container.
        chartContainerDiv.empty();

        // -------------------------------------------------------
        // 2. Append new chart containers, draw scatterplots.
        // -------------------------------------------------------

        let numDimensions   = this._data._model_metadata[this._data._modelID].n_components;
        let numScatterplots = ((numDimensions - 1) * (numDimensions)) / 2;

        // Generate all combinations of dimension indices.
        // todo Wrap scatterplots into rows in case of d > 2. Also consider case d == 1.
        for (let i = 0; i < numDimensions; i++) {
            let dataPaddingX = this._data._cf_intervals[i] * 0.1;

            for (let j = i + 1; j < numDimensions; j++) {
                let key             = i + ":" + j;
                let dataPaddingY    = this._data._cf_intervals[j] * 0.1;

                let scatterplotContainer = Utils.spawnChildDiv(
                    this._divStructure.scatterplotPaneID,
                    "model-detail-scatterplot-" + i + "-" + j,
                    "model-detail-scatterplot"
                );

                let scatterplot = dc.scatterPlot(
                    "#" + scatterplotContainer.id,
                    "model-detail-scatterplot-chart-group",
                    drMetaDataset,
                    null,
                    false
                );

                scatterplot
                    .height(chartContainerDiv.height() / numScatterplots - numScatterplots * 10)
                    .width(chartContainerDiv.width() / numScatterplots - numScatterplots * 10)
                    .useCanvas(true)
                    .x(d3.scale.linear().domain([
                        scope._data._cf_extrema[i].min - dataPaddingX,
                        scope._data._cf_extrema[i].max + dataPaddingX
                    ]))
                    .y(d3.scale.linear().domain([
                        scope._data._cf_extrema[j].min - dataPaddingY,
                        scope._data._cf_extrema[j].max + dataPaddingY
                    ]))
                    .renderHorizontalGridLines(true)
                    .dimension(scope._data._cf_dimensions[key])
                    .group(scope._data._cf_groups[key])
                    .keyAccessor(function(d) {
                        return d.key[0];
                     })
                    .valueAccessor(function(d) {
                        return d.key[1];
                     })
                    .existenceAccessor(function(d) {
                        return d.value.count > 0;
                    })
                    .excludedSize(2)
                    .excludedOpacity(0.7)
                    .excludedColor("#ccc")
                    .symbolSize(3)
                    // Filter on end of brushing action, not meanwhile (performance suffers otherwise).
                    .filterOnBrushEnd(true)
                    .mouseZoomable(true)
                    .margins({top: 25, right: 25, bottom: 25, left: 35});

                // Set number of ticks for y-axis.
                scatterplot.yAxis().ticks(3);
                scatterplot.xAxis().ticks(3);

                // Render chart.
                scatterplot.render();
            }
        }
    }

    /**
     * Draws sparklines for attributes (i. e. hyperparameters and objectives).
     * @private
     */
    _drawAttributeSparklines()
    {
        let dataset             = this._data;
        let drMetaDataset       = dataset._drMetaDataset;
        // Fetch metadata structure (i. e. attribute names and types).
        let metadataStructure   = drMetaDataset._metadata;

        // Fetch divs containing attribute sparklines.
        let hyperparameterContentDiv    = $("#" + this._divStructure.attributePane.hyperparameterContentID);
        let objectiveContentDiv         = $("#" + this._divStructure.attributePane.objectiveContentID);

        // Reset sparkline container div.
        hyperparameterContentDiv.html("");
        objectiveContentDiv.html("");

        // -------------------------------------------------------
        // 1. Gather/transform data.
        // -------------------------------------------------------

        // Gather values for bins from DRMetaDataset instance.
        let values = this._data.preprocessDataForSparklines();

        // -------------------------------------------------------
        // 2. Draw charts.
        // -------------------------------------------------------

        // Draw hyperparameter charts.
        for (let valueType in values) {
            for (let attribute of metadataStructure[valueType]) {
                let key     = valueType === "hyperparameters" ? attribute.name : attribute;
                let record  = values[valueType][key];

                // Append new div for attribute.
                let chartContainerDiv   = Utils.spawnChildDiv(
                    valueType === "hyperparameters" ? hyperparameterContentDiv[0].id : objectiveContentDiv[0].id,
                    null,
                    "model-detail-sparkline-container",
                    "<div class='attr-label'>" + DRMetaDataset.translateAttributeNames()[key] + "</div>"
                );
                let chartDiv            = Utils.spawnChildDiv(chartContainerDiv.id, null, "model-detail-sparkline");

                // Draw chart.
                $("#" + chartDiv.id).sparkline(
                    record.data,
                    {
                        type: "bar",
                        barWidth: Math.min(10, 50 / record.data.length),
                        barSpacing: 1,
                        chartRangeMin: 0,
                        height: 20,
                        tooltipFormat: '{{offset:offset}}',
                        tooltipValueLookups: {'offset': record.tooltips},
                        colorMap: record.colors
                    }
                );
            }
        }
    }

    processSettingsChange(delta)
    {
        // todo Which settings to consider?
    }

    /**
     * Updates dataset; re-renders charts.
     */
    update()
    {
        this._data      = this._operator._dataset;
        let data        = this._data;
        let stageDiv    = $("#" + this._operator._stage._target);

        // Show modal.
        $("#" + this._target).dialog({
            title: "Model Details for Model with ID #" + data._modelID,
            width: stageDiv.width() / 1.5,
            height: stageDiv.height() / 1.5
        });

        // Render charts.
        this.render();
    }
}