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

        // Generate table.
        let table = new Table(
            "Model selection table",
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
        return this._charts["Model selection table"];
    }
}