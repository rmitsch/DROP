import Panel from "./Panel.js";
import uuidv4 from "../utils.js";
import ParetoScatterplot from "../charts/ParetoScatterplot.js";

/**
 * Panel holding scatterplots and histograms in operator FilterReduce.
 */
export default class FilterReduceChartsPanel extends Panel
{
    /**
     * Constructs new FilterReduce charts panel.
     * @param name
     * @param operator
     * @param linked_crossfilter Reference to crossfilter instance. Might be null. If not null, specified crossfilter
     * instance is used (useful when multiple panels inside the same operator are supposed to operate on the same
     * dataset).
     */
    constructor(name, operator, linked_crossfilter)
    {
        super(name, operator, linked_crossfilter);

        // For all hyper-parameter/objective combinations: Construct one scatterplot.
        // Visual separation.
        // For all hyper-objective/objective combinations: Construct one scatterplot.


        let scatterplot = new ParetoScatterplot(
            "testscatterplot", this, ["n_components", "runtime"], this._crossfilter
        );

        // Render charts.
        dc.renderAll();

    }
}