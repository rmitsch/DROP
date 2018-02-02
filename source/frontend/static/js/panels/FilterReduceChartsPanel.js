import Panel from "./Panel.js";

/**
 * Panel holding scatterplots and histograms in operator FilterReduce.
 */
export default class FilterReduceChartsPanel extends Panel
{
    /**
     * Constructs new FilterReduce charts panel.
     * @param name
     * @param operator
     */
    constructor(name, operator)
    {
        super(name, operator);

        // For all hyper-parameter/objective combinations: Construct one scatterplot.
        // Visual separation.
        // For all hyper-objective/objective combinations: Construct one scatterplot.

        // Create a crossfilter instance.
        this._crossfilter = crossfilter(this._data);
        // CONTINUE HERE: Draw scatterplots, test brushing/linking.
        // Also: Style rules how (dynamic IDs)? Adaptions in terms of width/height, placement etc. probably necessary.
        // ndx -> crossfilter
    }
}