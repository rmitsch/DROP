import Panel from "./Panel.js";
import Utils from "../Utils.js";
import Dataset from "../Dataset.js";
import SurrogateModelChart from "../charts/SurrogateModelChart.js"


/**
 * Panel holding charts for surrogate model in SurrogateModelOperator.
 */
export default class SurrogateModelPanel extends Panel
{
    /**
     * Constructs new FilterReduce charts panel.
     * Note that no CF interaction happens in this panel - it's read only.
     * @param name
     * @param operator
     * @param parentDivID
     */
    constructor(name, operator, parentDivID)
    {
        super(name, operator, parentDivID);

        this._chart = null;

        // Update involved CSS classes.
        $("#" + this._target).addClass("surrogate-model-panel");

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
        console.log("Generating SurrogateModelPanel...");

        // Initialize chart.
        this._chart = new SurrogateModelChart(
            "Surrogate Model Chart",
            this,
            null,
            this._operator._dataset,
            null,
            this._divStructure.chartContainerID
        );
    }

    /**
     * Create (hardcoded) div structure for child nodes.
     * @returns {Object}
     * @private
     */
    _createDivStructure()
    {
        let scope = this;

        // -----------------------------------
        // Create chart container.
        // -----------------------------------

        let chartContainer = Utils.spawnChildDiv(this._target, null, "surrogate-model-chart-container");

        // -----------------------------------
        // Create title and options container.
        // -----------------------------------

        // Note: Listener for table icon is added by FilterReduceOperator, since it requires information about the table
        // panel.
        let infoDiv = Utils.spawnChildDiv(this._target, null, "surrogate-info");
        $("#" + infoDiv.id).html(
            "<span class='title'>" + scope._name + "</span>" +
            "<a id='surrogate-info-settings-icon' href='#'>" +
            "    <img src='./static/img/icon_settings.png' class='info-icon' alt='Settings' width='20px'>" +
            "</a>"
        )

        return {
            chartContainerID: chartContainer.id
        };
    }

    render()
    {
        console.log("rendering surrogatemodelpanel");
        this._chart.render();
    }
}