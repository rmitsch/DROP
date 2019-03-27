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

        this._sparklineValues = null;
        // Store information on split positions.
        this._lastSplitPositions    = {};
        this._splits                = {};

        // Update involved CSS classes.
        $("#" + this._target).addClass("model-detail-panel");

        // Create div structure for child nodes.
        this._divStructure = this._createDivStructure();

        // Dictionary for lookup of LIME rules.
        this._explanationRuleLookup = {};

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
     * Stores result in this._explanationRuleLookup.
     * @private
     */
    _updateExplanationRuleLookup()
    {
        for (let rule of this._data._preprocessedExplanationData) {
            if (!(rule.hyperparameter in this._explanationRuleLookup)) {
                this._explanationRuleLookup[rule.hyperparameter] = {};
            }

            this._explanationRuleLookup[rule.hyperparameter][rule.objective] = rule.hyperparameter + " to " + rule.objective + ": " + rule.weight;
        }
    }

    /**
     * Create (hardcoded) div structure for child nodes.
     * @returns {Object}
     */
    _createDivStructure()
    {
        let instance = this;

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
        this._splits["middle"] = Split(["#" + parameterPane.id, "#" + samplePane.id], {
            direction: "horizontal",
            sizes: [25, 75],
            onDragEnd: function() {
                instance.resize();
            }
        });
        this._lastSplitPositions["middle"] = [25, 75];

        // Split upper-left and bottom-left pane.
        this._splits["left"] = Split(["#" + attributePane.id, "#" + limePane.id], {
            direction: "vertical",
            sizes: [40, 60],
            onDragEnd: function() {
                instance.resize();
            }
        });
        this._lastSplitPositions["left"] = [40, 60];

        // Split upper-right and bottom-right pane.
        this._splits["right"] = Split(["#" + scatterplotPane.id, "#" + recordPane.id], {
            direction: "vertical",
            sizes: [50, 50],
            onDragEnd: function() {
                instance.resize();
            }
        });
        this._lastSplitPositions["right"] = [50, 50];

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

        this._redrawAttributeInfluenceHeatmap();

        // Update panel size.
        const panelDiv = $("#" + this._target);
        this._lastPanelSize = {width: panelDiv.width(), height: panelDiv.height()};
    }

    _redrawAttributeInfluenceHeatmap()
    {
        let scope       = this;
        let cfConfig    = this._data.crossfilterData["lime"];
        const attribute = "objective:hyperparameter";

        // Determine color scheme, color domain.
        const colorScheme = [
            // '#ca0020','#f4a582','#f7f7f7','#92c5de','#0571b0'
            // '#ca0020','#f4a582','#ffffff','#92c5de','#0571b0'
            '#a50f15', '#de2d26', '#fb6a4a', '#fcae91', '#fee5d9',
            "#ffffff", '#bdd7e7', '#6baed6', '#3182bd', '#08519c'
        ];
        // let colorDomain = ModelDetailPanel._calculateColorDomain(cfConfig.extrema["weight"], colorScheme);
        let colorDomain = ModelDetailPanel._calculateColorDomain({min: -1, max: 1}, colorScheme);

        const limePane = $("#model-details-lime-pane");
        this._charts["limeHeatmap"]
            .height(limePane.height() + 45)
            .width(limePane.width())
            .dimension(cfConfig.dimensions[attribute])
            .group(cfConfig.groups[attribute])
            .colorAccessor(d => d.value)
            .colors(
                d3.scale
                    .linear()
                    .domain(colorDomain)
                    .range(colorScheme)
            )
            .keyAccessor(d => d.key[0])
            .valueAccessor(d => d.key[1])
            .title(d => scope._explanationRuleLookup[d.key[1]][d.key[0]])
            .colsLabel(d => DRMetaDataset.translateAttributeNames(false)[d])
            .rowsLabel(d => DRMetaDataset.translateAttributeNames(false)[d])
            .margins({top: 0, right: 20, bottom: 48, left: 60})
            .transitionDuration(0)
            .xBorderRadius(0)
            // Rotate labels.
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

        this._charts["scatterplots"]    = {};
        const numDimensions             = this._data._allModelMetadata[this._data._modelID].n_components;
        let numPlotsInRow               = Math.max(numDimensions - 1, 1);
        const scatterplotWidth          = chartContainerDiv.width() / numPlotsInRow - 15;
        const scatterplotHeight         = chartContainerDiv.height() / numPlotsInRow - 10;

        // Generate all combinations of dimension indices.
        for (let i = 0; i < Math.max(numDimensions - 1, 1); i++) {
            let chartRowDiv = Utils.spawnChildDiv(
                this._divStructure.scatterplotPaneID,
                "model-detail-scatterplot-row-" + i,
                "model-detail-scatterplot-row"
            );
            $("#" + chartRowDiv.id).css('height', (scatterplotHeight + 5) + 'px');

            // Consider that we want to draw a scatterplot with the added "fake"/zero axis if we have a dataset with a
            // one-dim. embedding.
            for (let j = i + 1; j < Math.max(numDimensions, 2); j++) {
                // Generate scatterplot.
                let scatterplot = this._generateScatterplot(
                    [i, j],
                    {
                        height: scatterplotHeight,
                        width: scatterplotWidth
                    },
                    chartRowDiv.id
                );

                // Render chart.
                scatterplot.render();
                this._charts.scatterplots[i + ":" + j] = scatterplot;
            }
        }

        for (let scatterplotPos in this._charts.scatterplots) {
            this._setFilterHandler(this._charts.scatterplots[scatterplotPos], scatterplotPos);
        }
    }

    /**
     * Generates scatterplot (including divs).
     * @param currIndices Array of length 2 holding current indices (i, j). Used to generate keys for access to
     * crossfilter dimensions and groups and to generate unique div IDs.
     * @param scatterplotSize Size of scatterplot. Has .height and .width.
     * @param parentDivID
     * @returns {dc.scatterPlot} Generated scatter plt.
     * @private
     */
    _generateScatterplot(currIndices, scatterplotSize, parentDivID)
    {
        const numDimensions     = this._data._allModelMetadata[this._data._modelID].n_components;
        let cf_config           = this._data.crossfilterData["low_dim_projection"];
        const i                 = currIndices[0];
        const j                 = currIndices[1];
        const key               = i + ":" + j;
        let drMetaDataset       = this._operator._drMetaDataset;
        const dataPadding       = {
            x: cf_config.intervals[i] * 0.1,
            y: numDimensions > 1 ? cf_config.intervals[j] * 0.1 : 0.1
        };

        let scatterplotContainer = Utils.spawnChildDiv(
            parentDivID,
            "model-detail-scatterplot-" + i + "-" + j,
            "model-detail-scatterplot"
        );
        $("#" + scatterplotContainer.id).css('left', (i * (scatterplotSize.width + 10)) + 'px');

        let scatterplot = dc.scatterPlot(
            "#" + scatterplotContainer.id,
            this._target,
            drMetaDataset,
            null,
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
            .excludedSize(1)
            .excludedOpacity(0.7)
            .excludedColor("#ccc")
            .symbolSize(2)
            .filterOnBrushEnd(true)
            .mouseZoomable(true)
            .margins({top: 5, right: 0, bottom: 2, left: 0});

        scatterplot.yAxis().ticks(5);
        scatterplot.xAxis().ticks(5);

        return scatterplot;
    }

    /**
     * Sets chart's filter handler.
     * @param chart
     * @param pos
     * @private
     */
    _setFilterHandler(chart, pos)
    {
        let scatterplots = this._charts.scatterplots;

        chart.filterHandler(function (dimension, filters) {
            if (filters.length === 0) {
                dimension.filter(null);
                for (let scatterplotPos in scatterplots)
                    scatterplots[scatterplotPos].identifyFilteredRecords(d => true);
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

                let filteredIDs = new Set();
                for (const x in res) {
                    for (const y in res[x]) {
                        // https://stackoverflow.com/questions/32000865/simplest-way-to-merge-es6-maps-sets
                        filteredIDs = new Set(function*() {
                            yield* filteredIDs; yield* res[x][y].ids;
                        }());
                    }
                }
                dimension.filterFunction(d => filteredIDs.has(d[2]));

                for (let scatterplotPos in scatterplots) {
                    if (pos !== scatterplotPos)
                        scatterplots[scatterplotPos].identifyFilteredRecords(d => filteredIDs.has(d[2]));
                }
            }

            return filters;
        });
    }

    /**
     * Draws sparklines for attributes (i. e. hyperparameters and objectives).
     * @param updateValues
     * @private
     */
    _redrawAttributeSparklines(updateValues = true)
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
        if (updateValues || this._sparklineValues === false)
            this._sparklineValues = this._data.preprocessDataForSparklines();

        // -------------------------------------------------------
        // 2. Draw charts.
        // -------------------------------------------------------

        let attributeTable = "" +
            "<table style='width:100%' class='attributeTable'>" +
            "<tr>" +
                "<th>Name</th>" +
                "<th>Type </th>" +
                "<th>Value</th>" +
                "<th>Histogram</th>" +
            "</tr>";

        // Generate table for attribute charts.
        let chartDivIDs = {};
        for (const valueType in this._sparklineValues) {
            for (const attribute of metadataStructure[valueType]) {
                const key           = valueType === "hyperparameters" ? attribute.name : attribute;
                const value         = dataset._allModelMetadata[dataset._modelID][key];
                chartDivIDs[key]    = Utils.uuidv4();

                attributeTable += "<tr>";
                attributeTable +=   "<td>" + DRMetaDataset.translateAttributeNames()[key] + "</td>";
                attributeTable +=   "<td>" + (valueType === "hyperparameters" ? "HP" : "O") + "</td>";
                attributeTable +=   "<td>" + (isNaN(value) ? value : Math.round(value * 1000) / 1000) + "</td>";
                attributeTable +=   "<td id='" + chartDivIDs[key] + "'></td>";
                attributeTable += "</tr>";
            }
        }
        hyperparameterContentDiv.html(attributeTable + "</table>");

        // Draw hyperparameter charts.
        for (const valueType in this._sparklineValues) {
            for (const attribute of metadataStructure[valueType]) {
                let key             = valueType === "hyperparameters" ? attribute.name : attribute;
                let record          = this._sparklineValues[valueType][key];

                // Append new div for attribute.
                let chartContainerDiv = Utils.spawnChildDiv(
                    chartDivIDs[key], null, "model-detail-sparkline-container"
                );

                // Draw chart.
                const chartContainer = $("#" + chartContainerDiv.id);
                chartContainer.sparkline(
                    record.data,
                    {
                        type: "bar",
                        barWidth: Math.min(Math.max(10, chartContainer.width() / (record.data.length * 2)), 30),
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
        this._updateExplanationRuleLookup();

        // Show modal.
        $("#" + this._target).dialog({
            title: "Model Details for Model with ID #" + data._modelID,
            width: stageDiv.width() / 1.25,
            height: stageDiv.height() / 1.25,
            resizeStop: (event, ui) => this.resize()
        });

        // Render charts.
        this.render();
    }

    _updateTableHeight()
    {
        let tablePanelDiv = $("#model-details-block-record-table");
        $("div.model-detail-table .dataTables_scrollBody").css(
            'height', (tablePanelDiv.height() - 100) + "px"
        );
    }

    resize()
    {
        // Check modal.
        const panelDiv = $("#" + this._target);

        if (panelDiv.width() !== this._lastPanelSize.width || panelDiv.height() !== this._lastPanelSize.height) {
            // Update charts here if splits have been changed.

            this._lastPanelSize = {width: panelDiv.width(), height: panelDiv.height()};

            this._redrawAttributeSparklines(false);
            this._redrawAttributeInfluenceHeatmap();
            this._redrawRecordScatterplots();
            this._updateTableHeight();
        }

        else {
            // Check splits.
            for (const pos in this._splits) {
                const new_sizes = this._splits[pos].getSizes();

                if (new_sizes[0] !== this._lastSplitPositions[pos][0] || new_sizes[1] !== this._lastSplitPositions[pos][1]) {
                    this._lastSplitPositions[pos] = new_sizes;

                    if (pos === "left") {
                        this._redrawAttributeInfluenceHeatmap();
                    }
                    if (pos === "middle") {
                        this._redrawAttributeSparklines(false);
                        this._redrawAttributeInfluenceHeatmap();
                        this._redrawRecordScatterplots();
                    }
                    if (pos === "right") {
                        this._redrawRecordScatterplots();
                        this._updateTableHeight();
                    }
                }
            }
        }
    }

    highlight(id, source, propagate = false)
    {
        // We know that the only possible source we want to consider for a highlighting operation is the correponding
        // ModelDetailTable instance, so we can safely ignore all other sources.
        if (this._charts["table"].name === source) {
            for (const scatterplot in this._charts["scatterplots"])
                this._charts["scatterplots"][scatterplot].highlight(id);
        }
    }
}