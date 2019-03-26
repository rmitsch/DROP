import Utils from "../Utils.js";
import Dataset from "./Dataset.js";


/**
 * Wrapper class providing the specified dataset itself plus the corresponding crossfilter context and various utility
 * methods. */
export default class DRMetaDataset extends Dataset
{
    /**
     *
     * @param name
     * @param data Array of objects (JSON/array/dict/...) holding data to display. Note: Length of array defines number
     * of panels (one dataset per panel) and has to be equal with length of objects in metadata.
     * @param metadata Array of JSON objects holding metadata. Note: Length of array has to be equal with length of
     * data.
     * @param binCount Number of bins in histograms.
     */
    constructor(name, data, metadata, binCount)
    {
        super(name, data);

        this._dataIndicesByID   = {};
        this._metadata          = metadata;
        this._binCount          = 10;
        this._binCountSSP       = 10;
        this._correlations      = null;

        // Maps for translation of categorical variables into numerical ones.
        this._categoricalToNumericalValues = {};
        this._numericalToCategoricalValues = {};

        // Extract categorical hyperparameter for later shorthand usage.
        this._categoricalHyperparameterSet = this._extractCategoricalHyperparameters();

        // Update record metadata before further preprocessing.
        this._updateRecordMetadata();

        // Set up containers for crossfilter data.
        this._crossfilter = crossfilter(this._data);

        // Set up singular dimensions (one dimension per attribute).
        this._determineExtrema();

        // Set up histogram dimensions.
        this._initHistogramDimensionsAndGroups();

        // Set up binary dimensions (for scatterplots).
        this._initBinaryDimensionsAndGroups(true);

        // Compute correlation strengths.
        this.computeCorrelationStrengths();

        // Create series mapping.
        // Since for the intended use case (i. e. DROP) it is to be expected to need series variant w.r.t. each possible
        // hyperparameter, in makes sense to calculate all of them beforehand.
        this._seriesMappingByHyperparameter = this._generateSeriesMappingForHyperparameters();
    }

    /**
     * Prompts backend to compute correlations for currently filtered IDs.
     * @param callback
     */
    computeCorrelationStrengths(callback)
    {
        // Note that computing correlation for large datasets may take too long and thus correlation strengths are not
        // available before rendering starts. If this occurs, updating correlation strengths in UI have to be triggered
        // or request information periodically before UI is unlocked.
        let instance = this;

        if (this._cf_dimensions.id.top(Infinity).length > 1) {
            let idString = "";
            for (let embedding of this._cf_dimensions.id.top(Infinity))
                idString += embedding.id + ",";
            idString = idString.substring(0, idString.length - 1);

            fetch("/compute_correlation_strength?ids=" + idString,
                {
                    headers: {"Content-Type": "application/json; charset=utf-8"},
                    method: "GET"
                })
                .then(res => res.json())
                .then(function (results) {
                    instance._correlations = results;
                    if (callback !== undefined)
                        callback(results);
                });
        }

        else {
            callback(null);
        }
    }

    /**
     * Executes various tasks necessary before data can be preprocessed further.
     * Specifically:
     *  - Adds ID to individual records in this._data.
     *  - Updates this._metadata.hyperparameters[x].values with correct values as a workaround for bug in backend
     *    (transmttted hyperparameter values don't necessarily correspond with actual values, since they are hardcoded).
     * @private
     */
    _updateRecordMetadata()
    {
        // Store ID-to-index references for data elements.
        // Also: Collect hyperparameter values to update this._metadata.hyperparameters[x].values.
        let distinctHypValues = {};
        for (let hyp of this._metadata.hyperparameters)
            distinctHypValues[hyp.name] = new Set();
        for (let i = 0; i < this._data.length; i++) {
            this._dataIndicesByID[this._data[i].id] = i;
            for (let hyp of this._metadata.hyperparameters)
                distinctHypValues[hyp.name].add(this._data[i][hyp.name]);
        }
        // Update this._metadata.hyperparameters[x].values as workaround for bug in backend.
        for (let hyp of this._metadata.hyperparameters) {
            hyp.values = Array.from(distinctHypValues[hyp.name]).sort();
        }

        // Create numerical representations of categorical hyperparameters.
        this._discretizeCategoricalHyperparameters();
    }

    /**
     * Generates attribute-variant series for all hyperparameters.
     * Note that there are no predefined series for hyperparameter-based series, since they don't allow for natural
     * (i. e. with exactly one variant parameter) bindings. They could instead be connected by any number of common
     * properties, such as arbitrary hyperparameter settings, fuzzy value condition etc. (which are hence to be
     * calculate lazily on demand and on-the-fly).
     * @returns {{}}
     * @private
     */
    _generateSeriesMappingForHyperparameters()
    {
        let idToSeriesMappingByAttribute = {};

        // Loop through all hyperparameters, generate series for each of them.
        for (let attributeIndex in this._metadata.hyperparameters) {
            let variantAttribute = this._metadata.hyperparameters[attributeIndex].name;
            // Generate series for this variant attribute.
            idToSeriesMappingByAttribute[variantAttribute] = this._mapRecordsToSeries(variantAttribute);

            // If attribute is categorical: Also create series for its numerical representation.
            if (this._categoricalHyperparameterSet.has(variantAttribute))
                // Use already created series for categorical representation of this attribute.
                idToSeriesMappingByAttribute[variantAttribute + "*"] = idToSeriesMappingByAttribute[variantAttribute];
        }

        return idToSeriesMappingByAttribute;
    }

    /**
     * Maps records in this dataset to series w. r. t. to a invariant variable.
     * @param variantAttribute Attribute whose value is to be varied (while all others stay the same).
     * @returns {{}}
     * @private
     */
    _mapRecordsToSeries(variantAttribute)
    {
        let recordIDsToSeriesMap                = {};
        let seriesToRecordIDsMap                = {};
        let constantParameterSetsToSeriesMap    = {};
        let seriesCounter                       = 0;

        // Loop through all records.
        for (let record of this._data) {
            // Key holds stringified represenatation of constant parameters.
            let key = "";

            // Chain together key for this record.
            for (let attributeIndex in this._metadata.hyperparameters) {
                let attribute = this._metadata.hyperparameters[attributeIndex].name;
                if (attribute !== variantAttribute) {
                    key += record[this._metadata.hyperparameters[attributeIndex].name] + "_";
                }
            }
            key = key.slice(0, -1);

            // If key/constant parameter set doesn't exist yet: Create new series.
            if (!(key in constantParameterSetsToSeriesMap)) {
                // Link parameter set to series ID.
                constantParameterSetsToSeriesMap[key] = seriesCounter++;
                // Create new entry in map for linking series IDs to record IDs.
                seriesToRecordIDsMap[constantParameterSetsToSeriesMap[key]] = [];
            }
            // Link record ID to series ID.
            recordIDsToSeriesMap[record.id] = constantParameterSetsToSeriesMap[key];
            // Link series ID to IDs of records.
            seriesToRecordIDsMap[constantParameterSetsToSeriesMap[key]].push(record.id);
        }

        return {
            recordToSeriesMapping: recordIDsToSeriesMap,
            seriesToRecordMapping: seriesToRecordIDsMap,
            seriesCount: seriesCounter,
            variantAttribute: variantAttribute
        };
    }

    /**
     * Returns dict for translating column headers in JSON/dataframe into human-readable titles.
     * This is a catch-all for translation - all possible objectives and hyperparameters, regardless of the associated
     * DR algorithm, are included here for translation purposes.
     * @param useHTMLFormatting
     * @returns Dictionary with frontend translations for backend attributes.
     */
    static translateAttributeNames(useHTMLFormatting = true)
    {
        return {
            // Hyperparameters.
            "n_components": "Dimensions",
            "perplexity": "Perplexity",
            "early_exaggeration": "Early exagg.",
            "learning_rate": "Learning rate",
            "n_iter": "Iterations",
            "angle": "Angle",
            "metric": "Dist. metric",
            "metric*": "Dist. metric",
            "n_neighbors": "Neighbors",
            "min_dist": "Min. Distance",
            "local_connectivity": "Local Conn.",
            "n_epochs": "Iterations",
            // From here: Objectives.
            "r_nx": useHTMLFormatting ? "R<sub>nx</sub>" : "R_nx",
            "b_nx": useHTMLFormatting ? "B<sub>nx</sub>" : "B_nx",
            "stress": "Stress",
            "classification_accuracy": "RDP",
            "separability_metric": "Silhouette",
            "runtime": "Runtime"
        }
    }

    /**
     * Calculates extrema for specified dimensions/groups.
     * @param attribute
     * @param prefix
     * @param dataType "categorical" or "numerical". Distinction is necessary due to diverging structure of histogram
     * data.
     */
    _calculateExtremaForAttribute(attribute, prefix, dataType)
    {
        // Calculate extrema for histograms.
        let modifiedAttribute   = attribute + prefix;
        let sortedData          = JSON.parse(JSON.stringify(this._cf_groups[modifiedAttribute].all()));

        // Sort data by number of entries in this attribute's histogram.
        sortedData.sort(function(entryA, entryB) {
            let countA = dataType === "numerical" ? entryA.value.count : entryA.value;
            let countB = dataType === "numerical" ? entryB.value.count : entryB.value;

            return countA > countB ? 1 : (countB > countA ? -1 : 0);
        });

        // Determine extrema.
        this._cf_extrema[modifiedAttribute] = {
            min: ((dataType === "numerical") ? sortedData[0].value.count : sortedData[0].value),
            max: ((dataType === "numerical") ? sortedData[sortedData.length - 1].value.count : sortedData[sortedData.length - 1].value)
        };

        // Update extrema by padding values (hardcoded to 10%) for x-axis.
        this._cf_intervals[modifiedAttribute]   = this._cf_extrema[modifiedAttribute].max - this._cf_extrema[modifiedAttribute].min;
        if (this._axisPaddingRatio > 0) {
            this._cf_extrema[modifiedAttribute].min -= this._cf_intervals[modifiedAttribute] / this._axisPaddingRatio;
            this._cf_extrema[modifiedAttribute].max += this._cf_intervals[modifiedAttribute] / this._axisPaddingRatio;
        }
    }

    _determineExtrema()
    {
        let hyperparameterList  = Utils.unfoldHyperparameterObjectList(this._metadata.hyperparameters);
        let attributes          = JSON.parse(JSON.stringify(hyperparameterList.concat(this._metadata.objectives)));

        // -------------------------------------
        // Create dimension for ID.
        // -------------------------------------

        this._initSingularDimension("id");

        // -------------------------------------
        // Determine extrema and intervals.
        // -------------------------------------

        for (let catHP of this._categoricalHyperparameterSet)
            attributes.push(catHP + "*");

        // Initialize extrema.
        for (let i = 0; i < attributes.length; i++)
            this._cf_extrema[attributes[i]] = {max: -Infinity, min: Infinity};

        // Gather extrema.
        for (let record of this._data) {
            for (let attribute of attributes) {
                if (record[attribute] < this._cf_extrema[attribute].min)
                    this._cf_extrema[attribute].min = record[attribute];
                if (record[attribute] > this._cf_extrema[attribute].max)
                    this._cf_extrema[attribute].max = record[attribute];
            }
        }

        // Add padding.
        for (let attribute of attributes) {
            // Update extrema by padding values (hardcoded to 10%) for x-axis.
            this._cf_intervals[attribute] = this._cf_extrema[attribute].max - this._cf_extrema[attribute].min;

            // Add padding, if specified. Goal: Make also fringe elements clearly visible (other approach?).
            if (this._axisPaddingRatio > 0) {
                this._cf_extrema[attribute].min -= this._cf_intervals[attribute] / this._axisPaddingRatio;
                this._cf_extrema[attribute].max += this._cf_intervals[attribute] / this._axisPaddingRatio;
            }
        }
    }

    /**
     * Calculates singular extrema for given attribute, crossfilter dimension and crossfilter.
     * @param dimension
     * @param crossfilter
     * @returns {{extrema: {min, max}, interval: number}}
     * @private
     */
    _calculateSingularExtremaByAttribute_forCrossfilter(dimension, crossfilter)
    {
        // Calculate extrema for singular dimensions.
        let extrema = {
            min: dimension.bottom(1)[0].value,
            max: dimension.top(1)[0].value
        };

        // Update extrema by padding values (hardcoded to 10%) for x-axis.
        let interval = extrema.max - extrema.min;

        // Add padding, if specified. Goal: Make also fringe elements clearly visible (other approach?).
        if (this._axisPaddingRatio > 0) {
            extrema.min -= interval / this._axisPaddingRatio;
            extrema.max += interval / this._axisPaddingRatio;
        }

        return {extrema: extrema, interval: interval}
    }

    _initSingularDimension(attribute)
    {
        // Dimension with exact values.
        this._cf_dimensions[attribute] = this._crossfilter.dimension(
            function(d) { return d[attribute]; }
        );

        // Calculate extrema.
        this._calculateSingularExtremaByAttribute(attribute);
    }

    /**
     * Adds fields with binned values to record.
     * @param record
     * @param attributes
     * @param binnedFieldSuffix
     * @param binCount
     * @private
     */
    _addBinnedValuesToRecord(record, attributes, binnedFieldSuffix, binCount)
    {
        for (let i = 0; i < attributes.length; i++) {
            let attribute   = attributes[i];
            const binWidth  = this._cf_intervals[attribute] / binCount;
            const extrema   = this._cf_extrema[attribute];

            let value = record[attribute];
            if (value <= extrema.min)
                value = extrema.min;
            // Note: Removed `- binWidth` 2019-03-26 while fixing wrong bin computation for values in uppermost bin.
            else if (value >= extrema.max)
                value = extrema.max;

            // Adjust for extrema.
            let binnedValue = binWidth !== 0 ? Math.floor((value - extrema.min) / binWidth) * binWidth : 0;
            binnedValue += extrema.min;
            if (binnedValue > extrema.max)
                // Note: Removed `- binWidth` 2019-03-26 while fixing wrong bin computation for values in uppermost bin.
                binnedValue = extrema.max;

            record[attribute + binnedFieldSuffix] = binnedValue;
        }
    }

    /**
     * Initializes singular dimensions w.r.t. histograms.
     */
    _initHistogramDimensionsAndGroups() {
        let hyperparameters = Utils.unfoldHyperparameterObjectList(this._metadata.hyperparameters);
        let attributes = hyperparameters.concat(this.metadata.objectives);

        // Bin data for current attribute (i. e. hyperparameter or objective).
        for (let j = 0; j < this._data.length; j++) {
            // Add fields with binned values for histograms and SSPs.
            this._addBinnedValuesToRecord(this._data[j], attributes, "#histogram", this._binCount);
            this._addBinnedValuesToRecord(this._data[j], this.metadata.objectives, "#ssp", this._binCountSSP);
        }

        for (let i = 0; i < attributes.length; i++) {
            const attribute             = attributes[i];
            const histogramAttribute    = attributes[i] + "#histogram";

            // If this is a numerical hyperparameter or an objective: Returned binned width.
            if (i < hyperparameters.length &&
                this._metadata.hyperparameters[i].type === "numeric" ||
                i >= hyperparameters.length
            ) {
                // Dimension with rounded values (for histograms).
                this._cf_dimensions[histogramAttribute] = this._crossfilter.dimension(
                    function (d) {
                        return d[histogramAttribute];
                    }
                );

                // Create group for histogram.
                this._cf_groups[histogramAttribute] = this._generateGroupWithCounts(histogramAttribute);

                // Calculate extrema.
                this._calculateExtremaForAttribute(attribute, "#histogram", "numerical");
            }

            // Else if this is a categorical hyperparameter: Return value itself.
            else {
                this._cf_dimensions[histogramAttribute] = this._crossfilter.dimension(
                    function (d) {
                        return d[attribute];
                    }
                );

                // Create group for histogram.
                this._cf_groups[attribute + "#histogram"] = this._cf_dimensions[attribute + "#histogram"].group().reduceCount();

                // Calculate extrema.
                this._calculateExtremaForAttribute(attribute, "#histogram", "categorical");
            }
        }
    }

    /**
     * Initializes binary dimensions through cartesian product - one dimension per combination of
     * hyperparameter-objective and objective-objective pairings.
     * @param includeGroups Determines whether groups for binary dimensions should be generated as well.
     */
    _initBinaryDimensionsAndGroups(includeGroups = true)
    {
        // Transform list of hyperparameter objects into list of hyperparameter names.
        let hyperparameters             = Utils.unfoldHyperparameterObjectList(this._metadata.hyperparameters);
        let categoricalHyperparameters  = this._extractCategoricalHyperparameters();

        // Hyperparameter-objective and objective-objective pairings.
        for (let attribute1 of hyperparameters.concat(this._metadata.objectives)) {
            // Check if attribute is a categorical hyperparameter.
            // Use suffix "*" if attribute is categorical (and hence its numerical representation is to be used in
            // scatterplots).
            let processedAttribute1 = attribute1 + (categoricalHyperparameters.has(attribute1) ? "*" : "");

            for (let obj of this._metadata.objectives) {
                let combinedKey     = processedAttribute1 + ":" + obj;
                let transposedKey   = obj + ":" + processedAttribute1;

                // Only create new dimensions if transposed key didn't appear so far (i. e. the reverse combination
                // didn't already appear -> for A:B check if B:A was already generated).
                // Also: Drop auto-references (A:A).
                if (!(combinedKey in this._cf_dimensions) &&
                    !(transposedKey in this._cf_dimensions) &&
                    attribute1 !== obj
                ) {
                    // Create combined dimension (for scatterplot).
                    this._cf_dimensions[combinedKey] = this._crossfilter.dimension(
                        function(d) { return [d[processedAttribute1], d[obj + "#ssp"], d["id"]]; }
                    );

                    // Mirror dimension to transposed key.
                    this._cf_dimensions[transposedKey] = this._cf_dimensions[combinedKey];

                    // Create group for scatterplot.
                    this._cf_groups[combinedKey] = this._generateGroupWithCounts(combinedKey);

                    // Mirror group to transposed key.
                    this._cf_groups[transposedKey] = this._cf_groups[combinedKey];
                }
            }
        }
    }

    /**
     * Generates crossfilter group with information on number of elements..
     * @param attribute
     * @returns Newly generated group.
     * @private
     */
    _generateGroupWithCounts(attribute)
    {
        return this._cf_dimensions[attribute].group().reduce(
            function(elements, item) {
                elements.items.add(item);
                elements.ids.add(item.id);
                elements.count++;
                return elements;
            },
            function(elements, item) {
                elements.items.delete(item);
                elements.ids.delete(item.id);
                elements.count--;
                return elements;
            },
            function() {
                return { items: new Set(), count: 0, ids: new Set() };
            }
        );
    }

    /**
     * Discretizes all categorical hyperparameter. Manipulates specified list.
     * Adds necessary dimensions
     */
    _discretizeCategoricalHyperparameters()
    {
        // -------------------------------------------------
        // 1. Get metadata on categorical hyperparameters.
        // -------------------------------------------------

        for (let attributeIndex in this._metadata.hyperparameters) {
            if (this._metadata.hyperparameters[attributeIndex].type === "categorical") {
                let hyperparameterName = this._metadata.hyperparameters[attributeIndex].name;
                this._categoricalToNumericalValues[hyperparameterName] = {};
                this._numericalToCategoricalValues[hyperparameterName] = {};
            }
        }

        // -------------------------------------------------
        // 2. First pass: Get all values for cat. attributes.
        // -------------------------------------------------

        for (let i = 0; i < this._data.length; i++) {
            for (let param in this._categoricalToNumericalValues) {
                if (!(this._data[i][param] in this._categoricalToNumericalValues[param])) {
                    this._categoricalToNumericalValues[param][this._data[i][param]] = null;
                }
            }
        }

        // -------------------------------------------------
        // 3. Assign numerical reprentations based on
        // categories' ascending alphabetical order.
        // -------------------------------------------------

        // Use positive integer as numerical represenation.
        for (let param in this._categoricalToNumericalValues) {
            // Assign numerical representations in alphabetical order.
            let keys = Object.keys(this._categoricalToNumericalValues[param]).sort();
            for (let i = 0; i < keys.length; i++) {
                this._categoricalToNumericalValues[param][keys[i]]  = i + 1;
                this._numericalToCategoricalValues[param][i + 1]    = keys[i];
            }
        }

        // -------------------------------------------------
        // 4. Second pass: Add attributes for numerical
        // representation in dataset.
        // -------------------------------------------------

        // Suffix * is used to indiciate an attribute's numerical represenation.
        for (let i = 0; i < this._data.length; i++) {
            for (let param in this._categoricalToNumericalValues) {
                this._data[i][param + "*"] = this._categoricalToNumericalValues[param][this._data[i][param]];
            }
        }
    }

    /**
     * Fetches set of categorical hyperparameters' names.
     * @returns {Set<any>}
     * @private
     */
    _extractCategoricalHyperparameters()
    {
        let categoricalHyperparameters = new Set();
        for (let i = 0; i < this._metadata.hyperparameters.length; i++) {
            if (this._metadata.hyperparameters[i].type === "categorical")
                categoricalHyperparameters.add(this._metadata.hyperparameters[i].name);
        }

        return categoricalHyperparameters;
    }

    get metadata()
    {
        return this._metadata;
    }

    get crossfilter()
    {
        return this._crossfilter;
    }

    get cf_dimensions()
    {
        return this._cf_dimensions;
    }

    get cf_extrema()
    {
        return this._cf_extrema;
    }

    get cf_groups()
    {
        return this._cf_groups;
    }

    get categoricalToNumericalValues()
    {
        return this._categoricalToNumericalValues;
    }

    get numericalToCategoricalValues()
    {
        return this._numericalToCategoricalValues;
    }

    /**
     * Fetches record by its correspoding ID. Uses index structure to retrieve element from array.
     * @param recordID
     * @returns {*}
     */
    getDataByID(recordID)
    {
        return this._data[this._dataIndicesByID[recordID]];
    }

    /**
     * Restores object from instance string using cryo.js.
     * @param instanceString
     */
    static restoreFromString(instanceString)
    {
        let instance = Cryo.parse(window.name);
        Object.setPrototypeOf(instance, DRMetaDataset.prototype);
        Object.setPrototypeOf(instance._crossfilter, crossfilter.prototype);

        for (let groupname in instance._cf_groups) {

            console.log(instance._cf_groups[groupname].all());
        }

        return instance;
    }

    get seriesMappingByHyperparameter()
    {
        return this._seriesMappingByHyperparameter;
    }

    get correlations()
    {
        return this._correlations;
    }
}