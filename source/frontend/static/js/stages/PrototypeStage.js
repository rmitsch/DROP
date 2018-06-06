// // Import d3.js, crossfilter.js and dc.js.
// import * as d3 from "./static/lib/d3.v3";
// import * as crossfilter from "./static/lib/crossfilter.js";
// import * as dc from "./static/lib/dc.js";

// Import base class.
import Stage from './Stage.js'
import FilterReduceOperator from "../operators/FilterReduceOperator.js";
import SurrogateModelOperator from "../operators/SurrogateModelOperator.js";

/**
 * Stage for prototype (2018-02).
 */
export default class PrototypeStage extends Stage
{
    /**
     *
     * @param name
     * @param target ID of container div.
     * @param datasets Dictionary of isnstance of dataset class.
     */
    constructor(name, target, datasets)
    {
        super(name, target, datasets);

        // Construct operators.
        this.constructOperators()
    }

    /**
     * Construct all panels for this view.
     */
    constructOperators()
    {
        // Operators to be constructed:
        //  * Hyperparameter and objective selection.

        // --------------------------------
        // 1. Operator for hyperparameter and objective selection.
        // --------------------------------

        this.operators["FilterReduce"] = new FilterReduceOperator(
            "FilterReduce:TSNE",
            this,
            this._datasets["prototypeDataset"]
        );

        // --------------------------------
        // 2. Operators for exploration of surrogate model (read-only).
        // --------------------------------

        this.operators["SurrogateModel"] = new SurrogateModelOperator(
            "SurrogateModel:DecisionTree",
            this,
            this._datasets["prototypeDataset"],
            "Tree"
        );
    }


}