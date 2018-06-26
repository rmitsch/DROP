// // Import d3.js, crossfilter.js and dc.js.
// import * as d3 from "./static/lib/d3.v3";
// import * as crossfilter from "./static/lib/crossfilter.js";
// import * as dc from "./static/lib/dc.js";

// Import base class.
import Stage from './Stage.js'
import FilterReduceOperator from "../operators/FilterReduceOperator.js";
import SurrogateModelOperator from "../operators/SurrogateModelOperator.js";
import DissonanceOperator from "../operators/DissonanceOperator.js";
import Utils from "../Utils.js";

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

        // ---------------------------------------------------------
        // 1. Operator for hyperparameter and objective selection.
        // ---------------------------------------------------------

        // For panels at bottom: Spawn container.
        let splitTopDiv = Utils.spawnChildDiv(this._target, null, "split-top-container");

        this.operators["FilterReduce"] = new FilterReduceOperator(
            "FilterReduce:TSNE",
            this,
            this._datasets["modelMetadata"],
            splitTopDiv.id
        );

        // ---------------------------------------------------------
        // 2. Operator for exploration of surrogate model (read-only).
        // ---------------------------------------------------------

        // For panels at bottom: Spawn container. Used for surrogate and dissonance panel.
        let splitBottomDiv = Utils.spawnChildDiv(this._target, null, "split-bottom-container");

        // Q: How to get decision tree data?
        this._datasets["surrogateModel"] = null;

        this.operators["SurrogateModel"] = new SurrogateModelOperator(
            "GlobalSurrogateModel:DecisionTree",
            this,
            this._datasets["surrogateModel"],
            "Tree",
            splitBottomDiv.id
        );

        // ---------------------------------------------------------
        // 3. Operator for exploration of inter-model disagreement.
        // ---------------------------------------------------------

        // Q: How to get dissonance data? Has to be of pattern
        //      model.id -> sample.id, sample.value -> variance (or any other
        //      measure of disagreement to be used).
        this._datasets["dissonance"] = null;

        this.operators["Dissonance"] = new DissonanceOperator(
            "GlobalSurrogateModel:DecisionTree",
            this,
            this._datasets["dissonance"],
            splitBottomDiv.id
        );

        // ---------------------------------------------------------
        // 4. Initialize split panes.
        // ---------------------------------------------------------

        // Horizontal split.
        let surrTarget = this.operators["SurrogateModel"]._target;
        let dissTarget = this.operators["Dissonance"]._target
        $("#" + surrTarget).addClass("split split-horizontal");
        $("#" + dissTarget).addClass("split split-horizontal");
        Split(["#" + surrTarget, "#" + dissTarget], {
            direction: "horizontal",
            sizes: [50, 50]
        });

        // Vertical split.
        $("#" + splitTopDiv.id).addClass("split split-vertical");
        $("#" + splitBottomDiv.id).addClass("split split-vertical");
        Split(["#" + splitTopDiv.id, "#" + splitBottomDiv.id], {
            direction: "vertical",
            sizes: [55, 45]
        });
    }
}