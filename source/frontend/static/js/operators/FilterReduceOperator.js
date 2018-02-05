import Operator from "./Operator.js";
import FilterReduceChartsPanel from "../panels/FilterReduceChartsPanel.js";
import FilterReduceTablePanel from "../panels/FilterReduceTablePanel.js";

/**
 * Abstract base class for FilterReduce operators.
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
     * @param targetDivID
     */
    constructor(name, stage, dataset, targetDivID)
    {
        super(name, stage, "1", "n", dataset, targetDivID);

        // Construct all necessary panels.
        this.constructPanels();
    }

    /**
     * Constructs all panels required by this operator.
     */
    constructPanels()
    {
        // 1. Construct panels for charts.
        // let hypDiv = document.createElement('div');
        // hypDiv.id = 'hyperparameterObjectivesPanel';
        // hypDiv.className = 'panel';
        // $(this._target)[0].append(hypDiv);

        this.chartPanel = new FilterReduceChartsPanel(
            "Hyperparameters & Objectives",
            this
        );

        // 2. Construct panel for selection table.
        this.tablePanel = new FilterReduceTablePanel(
            "Model Selection",
            this
        );
    }
}