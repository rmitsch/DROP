import Scatterplot from "./Scatterplot.js";

/**
 * Scatterplots with dots connected by specified degree of freedom.
 */
export default class Scatterplot extends Scatterplot
{
    /**
     * Instantiates new ParetoScatterplot.
     * @param name
     * @param panel
     * @param attributes Attributes to be considered in plot. Has to be of length 2. First argument is projected onto
     * x-axis, second to y-axis. Attributes can contain one hyperparameter and one objective or two objectives.
     */
    constructor(name, panel, attributes)
    {
        super(name, panel, attributes);


        // Remaining todos for prototype:
        //      1. Duplicate scatterplot generated in FilterReduceChartsPanel here in ParetoScatterplot.
        //      2. Generate other scatterplots.
        //      3. Arrange layout + formatting + title(s).
        //      4. Generate histograms.
        //      5. Add other UI elements in panel (dropdown for connectBy()?).
        //      6. Model selection table.

        for (let attribute of this._attributes) {

        }
    }

    /**
     * Connects model parametrizations by specified attribute name. Used to e. g. draw pareto fronts.
     * @param hyperparameter Name of attribute to be used to connect models. Default value 'native' connects models with
     * diverging values for attribute on x-axis, but constant for all hyperparameter otherwise. Note that for
     * hyperparameter-objective plots all attributes other than native are ignored. Objective-objective plots accept all
     * available hyperparameters.
     */
    connectBy(hyperparameter = 'native')
    {
        throw new Error("ParetoScatterplot.connectBy(): Not implemented yet.");
    }

    render()
    {
        throw new Error("ParetoScatterplot.render(): Not implemented yet.");
    }
}