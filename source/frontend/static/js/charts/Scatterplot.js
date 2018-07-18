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
     * @param dataset
     * @param style Various style settings (chart width/height, colors, ...). Arbitrary format, has to be parsed indivdually
     * by concrete classes.
     * @param parentDivID
     * @param useBinning Defines whether (hexagonal) binning should be used.
     */
    constructor(name, panel, attributes, dataset, style, parentDivID, useBinning)
    {
        super(name, panel, attributes, dataset, style, parentDivID);

        // Check if attributes contain exactly two parameters.
        if (!Array.isArray(attributes) || attributes.length !== 2) {
            throw new Error("Scatterplot: Has to be instantiated with an array of attributes with length 2.");
        }

        // Construct dictionary for axis/attribute names.
        this._axes_attributes = {
            x: attributes[0],
            y: attributes[1]
        };
        // Store binning preference.
        this._useBinning = useBinning;


        // Construct graph.
        this.constructCFChart();
    }

    render()
    {
        throw new TypeError("Scatterplot.render(): Not implemented yet.");
    }

    constructCFChart()
    {
        throw new TypeError("Scatterplot.constructCFChart(): Not implemented yet.");
    }
}