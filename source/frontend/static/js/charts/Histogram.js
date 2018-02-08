import Chart from "./Chart.js";
import Utils from "../Utils.js"

/**
 * Creates histogram.
 */
export default class Histogram extends Chart
{
    /**
     *
     * @param name
     * @param panel
     * @param attributes
     * @param dataset
     * @param style
     * @param parentDivID
     */
    constructor(name, panel, attributes, dataset, style, parentDivID)
    {
        super(name, panel, attributes, dataset, style, parentDivID);

        // Check if attributes contain exactly two parameters.
        if (!Array.isArray(attributes) || attributes.length !== 1) {
            throw new Error("Histogram: Has to be instantiated with an array of attributes with length 1.");
        }

        // Construct dictionary for axis/attribute names.
        this._axes_attributes = {
            x: attributes[0]
        };

        // Construct graph.
        this.constructCFChart();
    }

    render()
    {
        throw new TypeError("Histogram.render(): Abstract method must not be called.");
    }

    constructCFChart()
    {
        throw new TypeError("Histogram.constructCFChart(): Abstract method must not be called.");
    }
}