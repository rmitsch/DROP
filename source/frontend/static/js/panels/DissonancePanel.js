import Panel from "./Panel.js";
import Utils from "../Utils.js";
import DissonanceChart from "../charts/DissonanceChart.js";

/**
 * Panel holding elements for comparison of inter-model disagreement for datapoints
 * in selected model instances.
 */
export default class DissonancePanel extends Panel
{
    /**
     * Constructs new panel for charts for DissonanceOperator.
     * @param name
     * @param operator
     * @param parentDivID
     */
    constructor(name, operator, parentDivID)
    {
        super(name, operator, parentDivID);

        // Update involved CSS classes.
        $("#" + this._target).addClass("dissonance-panel");

        // Create div structure for child nodes.
        this._divStructure = this._createDivStructure();

        // Generate charts.
        this._generateCharts();
    }

    /**
     * Generates all chart objects. Does _not_ render them.
     */
    _generateCharts()
    {
        console.log("Generating DissonancePanel...");

        // Initialize chart.
        this._chart = new DissonanceChart(
            "Sample Dissonance Chart",
            this,
            null,
            this._operator._dataset,
            null,
            this._divStructure.chartsContainerDivID
        );
        this._charts["histogramHeatmap"] = this._chart;
    }

    /**
     * Create (hardcoded) div structure for child nodes.
     * @returns {Object}
     */
    _createDivStructure()
    {
        let scope = this;

        // -----------------------------------
        // 1. Create charts container.
        // -----------------------------------

        let chartsContainerDiv  = Utils.spawnChildDiv(this._target, null, "dissonance-charts-container");

        // -----------------------------------
        // 2. Create title and options container.
        // -----------------------------------

        // Note: Listener for table icon is added by FilterReduceOperator, since it requires information about the table
        // panel.
        let infoDiv = Utils.spawnChildDiv(this._target, null, "panel-info");
        $("#" + infoDiv.id).html(
            "<span class='title'>" + scope._name + "</span>" +
            "<a id='dissonance-info-settings-icon' href='#'>" +
            "    <img src='./static/img/icon_settings.png' class='info-icon' alt='Settings' width='20px'>" +
            "</a>" +
            "<a id='dissonance-info-table-icon' href='#'>" +
            "    <img src='./static/img/icon_table.png' class='info-icon' alt='View in table' width='20px'>" +
            "</a>"
        );

        return {
            chartsContainerDivID: chartsContainerDiv.id
        };
    }

    render()
    {
        this._chart.render();
    }

    resize()
    {
        this._chart.resize();
    }

    processSettingsChange(delta)
    {
        this._chart.orderBy(delta);
    }
}