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
     * @param modelID
     * @param modelDataJSON Array of JSON objects holding model detail data.
     * @param drMetaDataset Reference to DRMetaDataset. Used to fetch data on embedding metadata and attribute metadata.
     */
    constructor(name, modelID, modelDataJSON, drMetaDataset)
    {
        super(name, modelDataJSON);

        // Update internal state.
        this._modelID               = modelID;
        this._drMetaDataset         = drMetaDataset;
        this._binCount              = drMetaDataset._binCount;
        this._low_dim_projection    = ModelDetailDataset._preprocessLowDimProjectionData(modelDataJSON.low_dim_projection);
        this._model_metadata        = modelDataJSON.model_metadata;
        this._originalDataset       = ModelDetailDataset._preprocessOriginalData(modelDataJSON.original_dataset);
        this._sampleDissonances     = modelDataJSON.sample_dissonances;

        // todo CONTINUE HERE:
        //  - Show data in model detail table.
        //  - Integration brushing linking between scatterplot/table.
        //  - Work on LIME integration.

        // Create crossfilter instance for low dimensoinal projection (LDP).
        this._ldp_crossfilter       = crossfilter(this._low_dim_projection);
        this._initLowDimProjectionCrossfilter();
    }

    /**
     * Converts low-dimensional projection data into a JSON object with ID and x_1...x_n coordinates.
     * @param coordinateLists
     * @private
     */
    static _preprocessLowDimProjectionData(coordinateLists)
    {
        let processedCoordinateObjects = [];

        for (let i = 0; i < coordinateLists.length; i++) {
            let newCoordinateObject = {id: i};
            for (let j = 0; j < coordinateLists[i].length; j++)
                newCoordinateObject[j] = coordinateLists[i][j];

            processedCoordinateObjects.push(newCoordinateObject)
        }

        return processedCoordinateObjects;
    }

    // scp vda-admin@bigmachine8.vda.univie.ac.at:/media/vda-admin/5e2ea268-53ea-410d-90d9-773e59974f94/drop .
    // /media/vda-admin/5e2ea268-53ea-410d-90d9-773e59974f94/drop

    /**
     * Updates structure of original data so it correlates with established structural convention.
     * @param originalData Dictionary containing original data.
     * @private
     */
    static _preprocessOriginalData(originalData)
    {
        let processedOriginalData = [];

        for (let key in originalData) {
            originalData[key].id = originalData[key].record_name;
            processedOriginalData.push(originalData[key]);
        }

        return processedOriginalData;
    }

    /**
     * Initializes all crossfilter-specific data used for low-dimensional projection/coordinates.
     * @private
     */
    _initLowDimProjectionCrossfilter()
    {
        // Create singular dimensions, determine extrema.
        this._initSingularDimensionsAndGroups();

        // Init binary dimensions and groups for scatterplot(s).
        this._initBinaryDimensionsAndGroups();
    }

    get data()
    {
        return this._data;
    }

    _initSingularDimensionsAndGroups()
    {
        let cf              = this._ldp_crossfilter;
        let numDimensions   = this._model_metadata[this._modelID].n_components;

        // Create singular dimensions.
        for (let i = 0; i < numDimensions; i++) {
            this._cf_dimensions[i] = cf.dimension(
                function(d) { return d[i]; }
            );
            // Calculate extrema.
            this._calculateSingularExtremaByAttribute(i);
        }
    }

    _initBinaryDimensionsAndGroups(includeGroups = true)
    {
        let cf              = this._ldp_crossfilter;
        let numDimensions   = this._model_metadata[this._modelID].n_components;

        // Generate groups for all combinations of dimension indices.
        for (let i = 0; i < numDimensions; i++) {
            for (let j = i + 1; j < numDimensions; j++) {
                let combinedKey     = i + ":" + j;
                let transposedKey   = j + ":" + i;

                // Create combined dimension (for scatterplot).
                this._cf_dimensions[combinedKey] = cf.dimension(
                    function(d) { return [d[i], d[j]]; }
                );
                // Mirror dimension to transposed key.
                this._cf_dimensions[transposedKey] = this._cf_dimensions[combinedKey];

                // Create group for scatterplot.
                this._cf_groups[combinedKey] = this._generateGroupWithCounts(combinedKey, [i, j]);
                // Mirror group to transposed key.
                this._cf_groups[transposedKey] = this._cf_groups[combinedKey];
            }
        }
    }

    _initSingularDimension(attribute)
    {
        throw new TypeError("ModelDetailDataset._initSingularDimension(): Abstract method must not be called.");
    }

    /**
     * Creates JSON object
     * @private
     */

    /**
     * Creates JSON object containing data preprocessed for usage in sparkline histograms - i. e. with filled gaps and
     * color/bin label encoding.
     * Note that presentation-specific encoding should actually happen in frontend.
     * @returns {{hyperparameters: {}, objectives: {}}}
     * @private
     */
    preprocessDataForSparklines()
    {
        let drMetaDataset       = this._drMetaDataset;
        // Fetch metadata structure (i. e. attribute names and types).
        let metadataStructure   = drMetaDataset._metadata;
        let currModelID         = this._modelID;

        // Gather values for bins from DRMetaDataset instance.
        let values = { hyperparameters: {}, objectives: {} };

        for (let valueType in values) {
            for (let attribute of metadataStructure[valueType]) {
                let key             = valueType === "hyperparameters" ? attribute.name : attribute;
                let unprocessedBins = JSON.parse(JSON.stringify(drMetaDataset._cf_groups[key + "#histogram"].all()));
                let binWidth        = drMetaDataset._cf_intervals[key] / drMetaDataset._binCount;
                // Determine whether this attribute is categorical.
                let isCategorical   = valueType === "hyperparameters" && attribute.type === "categorical";

                // Fill gaps with placeholder bins - we want empty bins to be respected in sparkline chart.
                // Only consider numerical values for now.
                let bins = isCategorical ? unprocessedBins : [];
                if (!isCategorical) {
                    for (let i = 0; i < drMetaDataset._binCount; i++) {
                        let currBinKey  = drMetaDataset._cf_extrema[key].min + binWidth * i;
                        let currBin     = unprocessedBins.filter(bin => { return bin.key === currBinKey; });

                        // Current bin not available: Create fake bin to bridge gap in chart.
                        bins.push(currBin.length < 1 ?
                            {
                                key: currBinKey,
                                value: {items: [], count: 0, extrema: {}}
                            } :
                            currBin[0]
                        );
                    }
                }

                // Build dict for this attribute.
                values[valueType][key]          = {data: [], extrema: drMetaDataset._cf_extrema[key], colors: null, tooltips: null};

                // Compile data list.
                values[valueType][key].data     = isCategorical ? bins.map(bin => bin.value) : bins.map(bin => bin.value.count);

                // Compile color map.
                values[valueType][key].colors   = bins.map(
                    bin => isCategorical ?
                    // If attribute is categorical: Check if bin key/title is equal to current model's attribute value.
                    (bin.key === this._model_metadata[currModelID][key] ? "red" : "#1f77b4") :
                    // If attribute is numerical: Check if list of items in bin contains current model with this ID.
                    bin.value.items.some(item => item.id === currModelID) ? "red" : "#1f77b4"
                );

                // Compile tooltip map.
                values[valueType][key].tooltips = {};
                for (let i = 0; i < bins.length; i++) {
                    values[valueType][key].tooltips[i] = isCategorical ?
                        bins[i].key :
                        bins[i].key.toFixed(4) + " - " + (bins[i].key + binWidth).toFixed(4);
                }
            }
        }

        return values;
    }
}