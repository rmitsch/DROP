import Operator from "./Operator.js";
import FilterReduceChartsPanel from "../panels/FilterReduceChartsPanel.js";
import FilterReduceTablePanel from "../panels/FilterReduceTablePanel.js";

/**
 * Class for FilterReduceOperators with scattered (scree) and violin plots.
 * One operator operates on exactly one dataset (-> one instance of class Dataset).
 */
export default class FilterReduceOperator extends Operator
{
    /**
     * Constructs new FilterReduceOperator.
     * Note that FilterReduceOperator is invariant towards which kernel was used for dimensionality reduction, since all
     * hyperparameter/objectives are defined in metadata. Thus, the specific operator name ("FilterReduce:TSNE") is
     * stored in a class attribute only (as opposed to branching the Operator class tree further).
     * @param name
     * @param stage
     * @param dataset Instance of Dataset class.
     * @param parentDivID
     */
    constructor(name, stage, dataset, parentDivID)
    {
        super(name, stage, "1", "n", dataset, parentDivID);

        // Update involved CSS classes.
        $("#" + this._target).addClass("filter-reduce-operator");

        // Construct all necessary panels.
        this.constructPanels();
    }

    /**
     * Constructs all panels required by this operator.
     */
    constructPanels()
    {
        // 1. Construct panels for charts.
        let frcPanel = new FilterReduceChartsPanel(
            "Hyperparameters & Objectives",
            this
        );
        this._panels[frcPanel.name] = frcPanel;

        // 2. Construct panel for selection table.
        let tablePanel = new FilterReduceTablePanel(
            "Model Selection",
            this
        );
        this._panels[tablePanel.name] = tablePanel;

        // 3. Set click listener for FRC panel's table modal.
        let scope = this;
        $("#filter-reduce-info-table-icon").click(function() {
            $("#" + scope._panels[tablePanel.name]._target).dialog({
                title: "All models",
                width: $("#" + scope._target).width() / 2,
                height: $("#" + scope._target).height()
            });
        });
    }
}