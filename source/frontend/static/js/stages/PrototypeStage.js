// // Import d3.js, crossfilter.js and dc.js.
// import * as d3 from "./static/lib/d3.v3";
// import * as crossfilter from "./static/lib/crossfilter.js";
// import * as dc from "./static/lib/dc.js";

// Import base class.
import Stage from './Stage.js'
import FilterReduceOperator from "../operators/FilterReduceOperator.js";

/**
 * Stage for prototype (2018-02).
 */
export default class PrototypeStage extends Stage
{
    /**
     *
     * @param name
     * @param target ID of container div.
     * @param data Array of objects (JSON/array/dict/...) holding data to display. Note: Length of array defines number
     * of panels (one dataset per panel) and has to be equal with length of objects in metadata.
     * @param metadata Array of JSON objects holding metadata. Note: Length of array has to be equal with length of
     * data.
     */
    constructor(name, target, data, metadata)
    {
        super(name, target, data, metadata);
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
            this._data,
            this._metadata
        );
    }
}