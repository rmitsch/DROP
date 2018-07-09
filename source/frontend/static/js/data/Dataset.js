/**
 * Abstract parent class for all datasets.
 */
export default class Dataset
{
    constructor(name, data)
    {
        this._name = name;
        this._data = data;

        this._cf_dimensions = {};
        this._cf_extrema    = {};
        this._cf_groups     = {};
        this._cf_intervals  = {};

        // Defines how much padding (relative to the shown interval) any axis should have.
        this._axisPaddingRatio  = 6.0;
    }

    get name()
    {
        return this._name;
    }

    get data()
    {
        return this._data;
    }

    /**
     * Initializes singular dimensions.
     * Note: Creates dimensions for all attribute by default. If not desired, columns have to be dropped beforehand.
     */
    _initSingularDimensionsAndGroups()
    {
        throw new TypeError("Dataset._initSingularDimensionsAndGroups(): Abstract method must not be called.");
    }

    /**
     * Initializes binary dimensions (for e. g. scatterplots or heatmaps).
     * @param includeGroups Determines whether groups for binary dimensions should be generated as well.
     */
    _initBinaryDimensionsAndGroups(includeGroups = true)
    {
        throw new TypeError("Dataset._initBinaryDimensionsAndGroups(): Abstract method must not be called.");
    }

    /**
     * Initializes singular dimensions and calculates extrema for specified attribute.
     * @param attribute
     * @private
     */
    _initSingularDimension(attribute)
    {
        throw new TypeError("Dataset._initSingularDimension(): Abstract method must not be called.");
    }

    /**
     * Calculates extrema for all singular dimensions.
     * @param attribute
     */
    _calculateSingularExtremaByAttribute(attribute)
    {
        // Calculate extrema for singular dimensions.
        this._cf_extrema[attribute] = {
            min: this._cf_dimensions[attribute].bottom(1)[0][attribute],
            max: this._cf_dimensions[attribute].top(1)[0][attribute]
        };
        // Update extrema by padding values (hardcoded to 10%) for x-axis.
        this._cf_intervals[attribute]   = this._cf_extrema[attribute].max - this._cf_extrema[attribute].min;

        // Add padding, if specified. Goal: Make also fringe elements clearly visible (other approach?).
        // todo *** METADATA SSP BUG: possible cause.
        if (this._axisPaddingRatio > 0) {
            this._cf_extrema[attribute].min -= this._cf_intervals[attribute] / this._axisPaddingRatio;
            this._cf_extrema[attribute].max += this._cf_intervals[attribute] / this._axisPaddingRatio;
        }
    }

    /**
     * Generates crossfilter group with information on number of elements..
     * @param attribute
     * @param primitiveAttributes List of relevenat attributes in original records. Extrema information is only
     * collected for these. Note of caution: Extrema are not to be considered reliable, since they aren't
     * updated after splicing operations (still sufficient for barchart highlighting operations though, since barchart/
     * group widths on x-axis don't change after splicing).
     * @returns Newly generated group.
     * @private
     */
    _generateGroupWithCounts(attribute, primitiveAttributes)
    {
        return this._cf_dimensions[attribute].group().reduce(
            function(elements, item) {
               elements.items.push(item);
               elements.count++;
               // Update extrema.
               for (let attr in elements.extrema) {
                   elements.extrema[attr].min = item[attr] < elements.extrema[attr].min ? item[attr] : elements.extrema[attr].min;
                   elements.extrema[attr].max = item[attr] > elements.extrema[attr].max ? item[attr] : elements.extrema[attr].max;
               }

               return elements;
            },
            function(elements, item) {
                elements.items.splice(elements.items.indexOf(item), 1);
                elements.count--;
                return elements;
            },
            function() {
                let extrema = {};
                for (let i = 0; i < primitiveAttributes.length; i++)
                    extrema[primitiveAttributes[i]] = {min: Number.MAX_VALUE, max: -Number.MAX_VALUE}

                return {items: [], count: 0, extrema: extrema};
            }
        );
}
}