import Panel from "./Panel.js";

/**
 * Panel holding table for selection of models in operator FilterReduce.
 */
export default class FilterReduceTablePanel extends Panel
{
    /**
     * Constructs new FilterReduce table panel.
     * @param name
     * @param operator
     */
    constructor(name, operator)
    {
        super(name, operator);

        // For all hyper-parameter/objective combinations: Construct one scatterplot.
        // Visual separation.
        // For all hyper-objective/objective combinations: Construct one scatterplot.


        // CONTINUE HERE: Draw scatterplots, test brushing/linking.
        // Also: Style rules how (dynamic IDs)? Adaptions in terms of width/height, placement etc. probably necessary.
        // ndx -> crossfilter
    }
}