import Operator from "./Operator.js";
import SurrogateModelPanel from "../panels/SurrogateModelPanel.js";


/**
 * Class for SurrogateModel operators.
 * One operator operates on exactly one dataset (-> one instance of class Dataset).
 */
export default class SurrogateModelOperator extends Operator
{
    /**
     * Constructs new SurrogateModelOperator.
     * Note that SurrogateModelOperators is invariant towards which kernel was used for dimensionality reduction, since
     * all hyperparameter/objectives are defined in metadata. Thus, the specific operator name ("SurrogateModel:Tree")
     * is stored in a class attribute only (as opposed to branching the Operator class tree further).
     * @param name
     * @param stage
     * @param dataset Instance of Dataset class.
     * @param modelType Type of model to be used as surrogate. Currently available: Decision tree.
     * @param parentDivID
     */
    constructor(name, stage, dataset, modelType, parentDivID)
    {
        super(name, stage, "1", "0", dataset, parentDivID);
        this._modelType = modelType;

        // Construct all necessary panels.
        this.constructPanels();
    }

    /**
     * Constructs all panels required by this operator.
     */
    constructPanels()
    {
        // Construct panel for surrogate model visualization.
        let surrModelPanel = new SurrogateModelPanel(
            "Surrogate Model",
            this
        );
        this._panels[surrModelPanel.name] = surrModelPanel;
    }

    // See https://bl.ocks.org/ajschumacher/65eda1df2b0dd2cf616f.
}