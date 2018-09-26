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
        this._modelID                   = modelID;
        this._drMetaDataset             = drMetaDataset;
        this._binCount                  = drMetaDataset._binCount;
        this._low_dim_projection        = ModelDetailDataset._preprocessLowDimProjectionData(
            modelDataJSON.low_dim_projection, modelDataJSON.original_dataset
        );
        this._model_metadata            = modelDataJSON.model_metadata;
        this._limeData                  = modelDataJSON.lime_explanation;
        this._sampleDissonances         = modelDataJSON.sample_dissonances;

        // Gather attributes available for original record.
        this._originalRecordAttributes  = [];
        for (let key in modelDataJSON.original_dataset[0]) {
            if (key !== "record_name")
                this._originalRecordAttributes.push(key);
        }

        //--------------------------------------
        // Initialize crossfilter datasets.
        //--------------------------------------

        this._crossfilterData = {};
        this._initCrossfilterData();
    }

    /**
     * Initializes crossfilter-related data.
     * @private
     */
    _initCrossfilterData()
    {
        for (let cf_dataset_name of ["low_dim_projection", "lime"]) {
            this._crossfilterData[cf_dataset_name] = {
                crossfilter: null,
                dimensions: {},
                groups: {},
                extrema: {},
                intervals: {}
            }
        }

        // Create crossfilter instance for low-dimensional projection (LDP).
        this._crossfilterData["low_dim_projection"].crossfilter = crossfilter(this._low_dim_projection);
        // Create crossfilter instance for LIME heatmap.
        this._crossfilterData["lime"].crossfilter               = crossfilter(ModelDetailDataset._preprocessLimeData(this._limeData));

        // Initialize dimensions and groups for crossfilter datasets.
        this._configureLowDimProjectionCrossfilter();
        this._configureLIMECrossfilter();
    }

    /**
     * Preprocesses LIME data to fit data pattern expected by crossfilter.js.
     * @param limeData
     * @returns {Array}
     * @private
     */
    static _preprocessLimeData(limeData)
    {
        let parsedLimeData = [];
        for (let objective in limeData) {
            for (let rule in limeData[objective]) {
                let original_rule = rule;
                let rule_parts = rule.split(" ");

                // Filter out hyperparameter. Consider that split/categorical arguments are structured differently.
                let hyperparameter = null;
                // Pattern: NUMBER OPERATOR HYPERPARAMETER OPERATOR NUMBER, e. g. 3 <= n_components < 6.
                if (rule_parts.length === 5)
                    hyperparameter = rule_parts[2];
                // Pattern: HYPERPARAMETER OPERATOR NUMBER, e. g. n_components > 2.
                else if (rule_parts.length === 3)
                    hyperparameter = rule_parts[0];
                // Pattern: HYPERPARAMETER=NUMBER, e. g. metric=cosine, indicating a categorical value.
                else if (rule_parts.length === 1) {
                    rule_parts = rule.split("=");

                    // Only accept hyperparameter if it was actually active in this record.
                    if (rule_parts[1] === "1") {
                        rule_parts = rule_parts[0].split("_");
                        hyperparameter = rule_parts[0];
                        rule = hyperparameter + " = " + rule_parts[1];
                    }
                }

                // Append parsed record, except when it contains a categorical value for this hyperparameter that
                // equates zero - meaning there is another record in the dataset with another categorical value for this
                // hyperparameter that actually applies.
                // Procedure to be debated - LIME values for these values seem to roughly cancel each other out, so as
                // an approximation it seems feasible.
                if (hyperparameter !== null)
                    parsedLimeData.push({
                        "objective": objective,
                        "hyperparameter": hyperparameter,
                        "rule":  rule,
                        "weight": limeData[objective][original_rule]
                    });
            }
        }

        return parsedLimeData;
    }

    /**
     * Configures dimensions and groups for LIME crossfilter used in heatmap.
     * @private
     */
    _configureLIMECrossfilter()
    {
        // todo Continue here: Create dimensions/groups necessary for heatmap.
        // Keep in mind that heatmap cells/labels have to be linked to rule data, incl. comparator;
        // while heatmap only shows rule weight.
        let config = this._crossfilterData.lime;
        console.log(ModelDetailDataset._preprocessLimeData(this._limeData));

        // Initialize binary dimension.
        config.dimensions["objective:hyperparameter"] = config.crossfilter.dimension(
            function(d) { return [d.objective, d.hyperparameter]; }
        );

        // Initialize group returning rule weight.
        config.groups["objective:hyperparameter"] = config.dimensions["objective:hyperparameter"].group().reduceSum(
            function(d) { return +d.weight; }
        );
    }

    /**
     * Converts low-dimensional projection data into a JSON object with ID and x_1...x_n coordinates.
     * Adds data from original records.
     * @param coordinateLists
     * @param originalData
     * @private
     */
    static _preprocessLowDimProjectionData(coordinateLists, originalData)
    {
        let processedCoordinateObjects = [];

        for (let i = 0; i < coordinateLists.length; i++) {
            let newCoordinateObject = {id: i};
            // Transform data into dict structure.
            for (let j = 0; j < coordinateLists[i].length; j++) {
                newCoordinateObject[j] = coordinateLists[i][j];
            }

            // If low-dim. projection is one-dimensional:
            // Pad coordinate list with second dimension with fixed values so that dataset can be shown in scatterplot
            // without further preprocessing.
            if (coordinateLists[i].length === 1)
                newCoordinateObject[1] = 0;

            // Append data from original records.
            for (let key in originalData[0]) {
                newCoordinateObject["orig_" + key] = originalData[i][key];
            }

            processedCoordinateObjects.push(newCoordinateObject)
        }

        return processedCoordinateObjects;
    }

    /**
     * Initializes all crossfilter-specific data used for low-dimensional projection/coordinates.
     * @private
     */
    _configureLowDimProjectionCrossfilter()
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
        let config          = this._crossfilterData["low_dim_projection"];
        let cf              = config.crossfilter;
        let numDimensions   = this._model_metadata[this._modelID].n_components;

        // Create singular dimensions.
        for (let i = 0; i < numDimensions; i++) {
            config.dimensions[i] = cf.dimension(
                function(d) { return d[i]; }
            );
            // Calculate extrema.
            let extremaInfo = this._calculateSingularExtremaByDimension(config.dimensions[i], i);
            config.extrema[i] = extremaInfo.extrema;
            config.intervals[i] = extremaInfo.interval;
        }

        // Create ID dimension.
        config.dimensions["id"] = cf.dimension(
            function(d) { return d.id; }
        );
    }

    _initBinaryDimensionsAndGroups(includeGroups = true)
    {
        let config          = this._crossfilterData["low_dim_projection"];
        let cf              = config.crossfilter;
        let numDimensions   = this._model_metadata[this._modelID].n_components;

        // Generate groups for all combinations of dimension indices.
        for (let i = 0; i < numDimensions; i++) {
            // Consider that "fake" coordinate in 1D projections has to be part of a binary dim./group as well.
            for (let j = i + 1; j < (numDimensions > 1 ? numDimensions : 2); j++) {
                let combinedKey     = i + ":" + j;
                let transposedKey   = j + ":" + i;

                // Create combined dimension (for scatterplot).
                config.dimensions[combinedKey] = cf.dimension(
                    function(d) { return [d[i], d[j]]; }
                );
                // Mirror dimension to transposed key.
                config.dimensions[transposedKey] = config.dimensions[combinedKey];

                // Create group for scatterplot.
                config.groups[combinedKey] = this._generateGroupWithCountsForDimension(
                    config.dimensions[combinedKey], [i, j]
                );
                // Mirror group to transposed key.
                config.groups[transposedKey] = config.groups[combinedKey];
            }
        }
    }

    _initSingularDimension(attribute)
    {
        throw new TypeError("ModelDetailDataset._initSingularDimension(): Abstract method must not be called.");
    }

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

    get crossfilterData()
    {
        return this._crossfilterData;
    }
}