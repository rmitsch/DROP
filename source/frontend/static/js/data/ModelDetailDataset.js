import Utils from "../Utils.js";
import Dataset from "./Dataset.js";


/**
 * Class containing data and methods for one individual DR model with all relevant details, i. e. records' labels,
 * classes, coordinates etc.
 */
export default class ModelDetailDataset extends Dataset
{
    /**
     * Initializes new ModelDetailDataset.
     * @param name
     * @param data Array of objects (JSON/array/dict/...) holding data to display. Note: Length of array defines number
     * of panels (one dataset per panel) and has to be equal with length of objects in metadata.
     * @param modelDataJSON Array of JSON objects holding model detail data.
     * @param binCount Number of bins in histograms.
     */
    constructor(name, modelDataJSON, binCount)
    {
        super(name, modelDataJSON);
    }

    get data()
    {
        return this._data;
    }

    _initSingularDimensionsAndGroups()
    {
        throw new TypeError("Dataset._initSingularDimensionsAndGroups(): Abstract method must not be called.");
    }

    _initBinaryDimensionsAndGroups(includeGroups = true)
    {
        throw new TypeError("Dataset._initBinaryDimensionsAndGroups(): Abstract method must not be called.");
    }

    _initSingularDimension(attribute)
    {
        throw new TypeError("Dataset._initSingularDimension(): Abstract method must not be called.");
    }
}