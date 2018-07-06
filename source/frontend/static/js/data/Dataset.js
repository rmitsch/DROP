/**
 * Abstract parent class for all datasets.
 */
export default class Dataset
{
    constructor(name, data)
    {
        this._name = name;
        this._data = data;
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
}