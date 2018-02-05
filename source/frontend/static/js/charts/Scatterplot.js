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
     * @param crossfilter
     */
    constructor(name, panel, attributes, crossfilter)
    {
        super(name, panel, attributes, crossfilter);

                // Check if attributes contain exactly two parameters.
        if (!Array.isArray(attributes) || attributes.length !== 2) {
            throw new Error("ParetoScatterplot: Has to be instantiated with an array of attributes with length 2.");
        }

        // Store reference to this instance for later use in nested expressions.
        let instance = this;

        // Construct dictionary for axis/attribute names.
        this._axes_attributes = {
            x: attributes[0],
            y: attributes[1]
        };

        // Construct dictionary for crossfilter dimensions.
        this._cf_dimensions = {
            x: this._crossfilter.dimension(function(d) { return d[instance._axes_attributes.x]; }),
            y: this._crossfilter.dimension(function(d) { return d[instance._axes_attributes.y]; }),
            // Definition of 2D-dimension necessary for dc.js' scatterplot object.
            total: this._crossfilter.dimension(function(d) {
                return [d[instance._axes_attributes.x], d[instance._axes_attributes.y]];
            })
        };

        // Calculate extrema. Use predefined constant for generating a small padding between actual extrema and chart
        // border.
        this._cf_extrema = {
            x: {
                min: this._cf_dimensions.x.bottom(1)[0][this._axes_attributes.x] * 0.9,
                max: this._cf_dimensions.x.top(1)[0][this._axes_attributes.x] * 1.1
            },
            y: {
                min: this._cf_dimensions.y.bottom(1)[0][this._axes_attributes.y] * 0.9,
                max: this._cf_dimensions.y.top(1)[0][this._axes_attributes.y] * 1.1
            }
        };

        // Define group aggregations.
        this._cf_groups = {
            total: this._cf_dimensions.total.group().reduce(
                function(elements, item) {
                    elements.items.push(item);
                    elements.count++;

                    return elements;
                },
                function(elements, item) {
                    elements.items.splice(elements.items.indexOf(item), 1);
                    elements.count--;

                    return elements;
                },
                function() {
                    return {items: [], count: 0};
                }
            )
        };

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