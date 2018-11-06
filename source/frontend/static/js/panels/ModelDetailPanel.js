import Panel from "./Panel.js";
import Utils from "../Utils.js";
import DRMetaDataset from "../data/DRMetaDataset.js";
import ModelDetailTable from "../charts/ModelDetailTable.js";

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

        // Dictionary for lookup of LIME rules.
        this._limeRuleLookup = {};

        // Generate charts.
        this._generateCharts();
    }

    /**
     * Calculates color domain based on existing color scheme and data extrema.
     * @param extrema
     * @param colorScheme
     * @returns {number[]}
     * @private
     */
    static _calculateColorDomain(extrema, colorScheme)
    {
        let colorDomain = [];

        colorScheme.forEach(
            (color, i) => colorDomain.push(extrema.min + ((extrema.max - extrema.min) / colorScheme.length * i))
        );

        return colorDomain;
    }

    /**
     * Generates all chart objects. Does _not_ render them.
     */
    _generateCharts()
    {
        console.log("Generating ModelDetailPanel...");

        // Use operator's target ID as group name.
        let dcGroupName = this._operator._target;

        // Initialize table.
        this._charts["table"] = null;

        // Initialize LIME heatmap.
        this._charts["limeHeatmap"] = dc.heatMap(
            "#" + this._divStructure.limePaneID,
            dcGroupName
        );
    }

    /**
     * Extracts dictionary {hyperparameter -> {objective -> rule}} from loaded dataset.
     * Stores result in this._limeRuleLookup.
     * @private
     */
    _updateLimeRuleLookup()
    {
        for (let rule of this._data._preprocessedLimeData) {
            if (!(rule.hyperparameter in this._limeRuleLookup)) {
                this._limeRuleLookup[rule.hyperparameter] = {};
            }

            this._limeRuleLookup[rule.hyperparameter][rule.objective] = rule.rule + ": " + rule.weight;
        }
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
                <div id="model-details-lime-pane"</div>
            </div>`
        );

        // 3. Upper-right pane - all records in scatterplot (SPLOM? -> What to do with higher-dim. projections?).
        let scatterplotPane = Utils.spawnChildDiv(
            samplePane.id, null, "model-detail-pane split-vertical",
            `<div class='model-details-block reduced-padding'>
                <div class='model-details-title'>All Records</div>
            </div>`
        );

        // 4. Bottom-right pane - detailed information to currently selected record.
        let recordPane = Utils.spawnChildDiv(
            samplePane.id, null, "model-detail-pane split-vertical",
            `<div class='model-details-block' id='model-details-block-record-table'>
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
            sizes: [40, 60],
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
            limePaneID: "model-details-lime-pane",
            scatterplotPaneID: scatterplotPane.id,
            recordPane: {
                id: recordPane.id,
                tableID: "model-details-block-record-table"
            }
        };
    }

    render()
    {
        // -------------------------------------------------------
        // 1. Draw sparklines for attributes.
        // -------------------------------------------------------

        this._redrawAttributeSparklines();

        // -------------------------------------------------------
        // 2. Draw scatterplot/SPLOM showing individual records.
        // -------------------------------------------------------

        this._redrawRecordScatterplots();

        // -------------------------------------------------------
        // 3. Update table.
        // -------------------------------------------------------

        this._reconstructTable();

        // -------------------------------------------------------
        // 4. Draw LIME matrix.
        // -------------------------------------------------------

        this._redrawLIMEHeatmap();
    }

    _redrawLIMEHeatmap()
    {
        let scope       = this;
        let cfConfig    = this._data.crossfilterData["lime"];
        let attribute   = "objective:hyperparameter";

        // Determine color scheme, color domain.
        let colorScheme = [
            '#ca0020','#f4a582','#f7f7f7','#92c5de','#0571b0'
        ];
        // let colorDomain = ModelDetailPanel._calculateColorDomain(cfConfig.extrema["weight"], colorScheme);
        let colorDomain = ModelDetailPanel._calculateColorDomain({min: -1, max: 1}, colorScheme);

        this._charts["limeHeatmap"]
            .height($("#model-details-lime-pane").height())
            .width($("#model-details-lime-pane").width())
            .dimension(cfConfig.dimensions[attribute])
            .group(cfConfig.groups[attribute])
            .colorAccessor(function(d)  { return d.value; })
            .colors(
                d3.scale
                    .linear()
                    .domain(colorDomain)
                    .range(colorScheme)
            )
            .keyAccessor(function(d)    { return d.key[0]; })
            .valueAccessor(function(d)  { return d.key[1]; })
            .title(function(d) {
                return scope._limeRuleLookup[d.key[1]][d.key[0]];
            })
            .colsLabel(function(d)      { return DRMetaDataset.translateAttributeNames(false)[d]; })
            .rowsLabel(function(d)      { return DRMetaDataset.translateAttributeNames(false)[d]; })
            .margins({top: 0, right: 20, bottom: 48, left: 60})
            .transitionDuration(0)
            .xBorderRadius(0)
            // Rotrate labels.
            .on('pretransition', function(chart) {
                chart
                    .selectAll('g.cols.axis > text')
                    .attr('transform', function (d) {
                        let coord = this.getBBox();
                        let x = coord.x + (coord.width/2) + coord.height * 1.5,
                            y = coord.y + (coord.height/2) * 5;

                        return "rotate(-50 "+ x + " " + y + ")"
                    });
            });
        this._charts["limeHeatmap"].render();
    }

    _reconstructTable()
    {
        // Remove old table, if exists.
        $('div.model-detail-table').remove();

        // Construct new table - easier than updating existing one.
        this._charts["table"] = new ModelDetailTable(
            "Model Detail ModelOverviewTable",
            this,
            this._data._originalRecordAttributes,
            this._data,
            null,
            this._divStructure.recordPane.tableID
        );
    }

    _redrawRecordScatterplots()
    {
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

        let numDimensions   = this._data._allModelMetadata[this._data._modelID].n_components;
        let numScatterplots = ((numDimensions - 1) * (numDimensions)) / 2;

        // Generate all combinations of dimension indices.
        for (let i = 0; i < numDimensions; i++) {
            // Consider that we want to draw a scatterplot with the added "fake"/zero axis if we have a dataset with a
            // one-dim. embedding.
            for (let j = i + 1; j < (numDimensions > 1 ? numDimensions : 2); j++) {
                // Generate scatterplot.
                let scatterplot = this._generateScatterplot(
                    [i, j],
                    {
                        height: chartContainerDiv.height() / numScatterplots - numScatterplots * 10,
                        width: chartContainerDiv.width() / numScatterplots - numScatterplots * 10
                    }
                );

                // Render chart.
                scatterplot.render();
            }
        }
    }

    /**
     * Generates scatterplot (including divs).
     * @param currIndices Array of length 2 holding current indices (i, j). Used to generate keys for access to
     * crossfilter dimensions and groups and to generate unique div IDs.
     * @param scatterplotSize Size of scatterplot. Has .height and .width.
     * @returns {dc.scatterPlot} Generated scatter plt.
     * @private
     */
    _generateScatterplot(currIndices, scatterplotSize)
    {
        let cf_config           = this._data.crossfilterData["low_dim_projection"];
        let i                   = currIndices[0];
        let j                   = currIndices[1];
        let key                 = i + ":" + j;
        let scope               = this;
        let drMetaDataset       = this._operator._drMetaDataset;
        let dataPadding         = {
            x: cf_config.intervals[i] * 0.1,
            y: cf_config.intervals[j] * 0.1
        };

        let scatterplotContainer = Utils.spawnChildDiv(
            this._divStructure.scatterplotPaneID,
            "model-detail-scatterplot-" + i + "-" + j,
            "model-detail-scatterplot"
        );

        let scatterplot = dc.scatterPlot(
            "#" + scatterplotContainer.id,
            this._target,
            drMetaDataset,
            null,
            false
        );

        // Render scatterplot.
        scatterplot
            .height(scatterplotSize.height)
            .width(scatterplotSize.width)
            .useCanvas(true)
            .x(d3.scale.linear().domain([
                cf_config.extrema[i].min - dataPadding.x,
                cf_config.extrema[i].max + dataPadding.x
            ]))
            .y(d3.scale.linear().domain([
                cf_config.extrema[j].min - dataPadding.y,
                cf_config.extrema[j].max + dataPadding.y
            ]))
            .renderHorizontalGridLines(true)
            .renderVerticalGridLines(true)
            .dimension(cf_config.dimensions[key])
            .group(cf_config.groups[key])
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

        return scatterplot;
    }

    /**
     * Draws sparklines for attributes (i. e. hyperparameters and objectives).
     * @private
     */
    _redrawAttributeSparklines()
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
     * Updates dataset; re-renders charts with new data.
     */
    update()
    {
        this._data      = this._operator._dataset;
        let data        = this._data;
        let stageDiv    = $("#" + this._operator._stage._target);

        // Update LIME rule lookup.
        this._updateLimeRuleLookup();

        // Show modal.
        $("#" + this._target).dialog({
            title: "Model Details for Model with ID #" + data._modelID,
            width: stageDiv.width() / 1.25,
            height: stageDiv.height() / 1.25
        });

        // Render charts.
        this.render();
    }
}