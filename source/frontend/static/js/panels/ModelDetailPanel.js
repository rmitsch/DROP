import Panel from "./Panel.js";
import Utils from "../Utils.js";
import Table from "../charts/Table.js"
import DissonanceChart from "../charts/DissonanceChart";

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

        // Generate charts.
        this._generateCharts();
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
        let scope = this;

        // -----------------------------------
        // 1. Create charts container.
        // -----------------------------------
        //
        // let chartsContainerDiv  = Utils.spawnChildDiv(this._target, null, "dissonance-charts-container");
        //
        // // -----------------------------------
        // // 2. Create title and options container.
        // // -----------------------------------
        //
        // // Note: Listener for table icon is added by FilterReduceOperator, since it requires information about the table
        // // panel.
        // let infoDiv = Utils.spawnChildDiv(this._target, null, "dissonance-info");
        // $("#" + infoDiv.id).html(
        //     "<span class='title'>" + scope._name + "</span>" +
        //     "<a id='dissonance-info-settings-icon' href='#'>" +
        //     "    <img src='./static/img/icon_settings.png' class='info-icon' alt='Settings' width='20px'>" +
        //     "</a>" +
        //     "<a id='dissonance-info-table-icon' href='#'>" +
        //     "    <img src='./static/img/icon_table.png' class='info-icon' alt='View in table' width='20px'>" +
        //     "</a>"
        // )
        //
        // return {
        //     chartsContainerDivID: chartsContainerDiv.id
        // };
    }

    render()
    {
        // this._chart.render();
    }

    processSettingsChange(delta)
    {
        // this._chart.orderBy(delta);
    }
}