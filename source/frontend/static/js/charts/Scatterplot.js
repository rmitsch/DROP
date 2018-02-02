import Chart from "./Chart.js";

/**
 * Base class for scatterplots.
 */
export default class Scatterplot extends Chart
{
    /**
     * Instantiates new scatter plot.
     * @param name
     * @param panel
     * @param attributes Attributes that are to be considered in this chart (how exactly is up to the implementation of
     * the relevant subclass(es)).
     */
    constructor(name, panel, attributes)
    {
        super(name, panel, attributes);
    }

    render()
    {
        throw new TypeError("Scatterplot.render(): Not implemented yet.");
    }
}