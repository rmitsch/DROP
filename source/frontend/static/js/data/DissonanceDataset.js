import Utils from "../Utils.js";
import Dataset from "./Dataset.js";

/**
 * Class holding raw and processed data for sample-in-model dissonance data.
 */
export default class DissonanceDataset extends Dataset
{
    constructor(name, data)
    {
        super(name, data);

        console.log(data);

        // todo
        // * Sample ID to sample name mapping
        // * UI: Search field
        // * Create dimensions and groups.
        // * Integrate DissonanceDataset in DissonanceChart.

        // Set up containers for crossfilter data.
        this._crossfilter   = crossfilter(this._data);
        this._cf_dimensions = {};
        this._cf_extrema    = {};
        this._cf_groups     = {};
        this._cf_intervals  = {};

        // Initialize crossfilter data.
        this._initSingularDimensionsAndGroups();
        this._initBinaryDimensionsAndGroups();
    }

    _initSingularDimensionsAndGroups()
    {
    }

    _initBinaryDimensionsAndGroups()
    {
        this._cf_dimensions["heatmap"] = this._crossfilter.dimension(
            function(d) {
                return [d["sample_id"], d["model_id"]];
            }
        );
    }

    _initSingularDimension(attribute)
    {
    }
}