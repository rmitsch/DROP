import Panel from "./Panel.js";
import Utils from "../Utils.js";
import Table from "../charts/ModelOverviewTable.js"

/**
 * Panel holding table for selection of models in operator FilterReduce.
 */
export default class FilterReduceTablePanel extends Panel
{
    /**
     * Constructs new FilterReduce table panel.
     * @param name
     * @param operator
     * @param parentDivID
     */
    constructor(name, operator, parentDivID)
    {
        super(name, operator, parentDivID);

        // Update involved CSS classes.
        $("#" + this._target).addClass("filter-reduce-table-panel");

        // Initialize div structure.
        this._divStructure = this._createDivStructure();

        // Generate table.
        this._tableName = "Model selection table";
        let table = new Table(
            this._tableName,
            this,
            Utils.unfoldHyperparameterObjectList(
                this._operator.dataset.metadata.hyperparameters
            ).concat(this._operator.dataset.metadata.objectives),
            this._operator.dataset,
            null,
            this._target
        );
        this._charts[table.name] = table;
    }

    get table()
    {
        return this._charts[this._tableName];
    }

    /**
     * Create (hardcoded) div structure for child nodes.
     * @returns {Object}
     */
    _createDivStructure()
    {
        let scope   = this;
        let infoDiv = Utils.spawnChildDiv(this._target, null, "panel-info");

        $("#" + infoDiv.id).html(
            "<span class='title'>All embeddings</span>" +
            "<a id='filter-reduce-table-info-reset-icon' href='#'>" +
            "    <img src='./static/img/icon_reset_2.png' class='info-icon' alt='Reset' width='19px'>" +
            "</a>" +
            "<a id='filter-reduce-table-info-settings-icon' href='#'>" +
            "    <img src='./static/img/icon_settings.png' class='info-icon' alt='Settings' width='20px'>" +
            "</a>"
        );


        // Set listener for reset icon.
        $("#filter-reduce-table-info-reset-icon").click(function() {
            scope._charts[scope._tableName].resetFilterStatus();
        });

        return {
          infoDivID: infoDiv.id
        };
    }

    resize()
    {
        let panelDiv = $("#" + this._target);
        if (panelDiv.height() !== this._lastPanelSize.height) {
            $("#" + this._target + " .dataTables_scrollBody").css(
                'height', (panelDiv.height() - 190) + "px"
            );
        }

        // Store size of panel at time of last render.
        this._lastPanelSize.width = panelDiv.width();
        this._lastPanelSize.height = panelDiv.height();
    }

    render()
    {
        this._charts[this._tableName].render();
    }
}