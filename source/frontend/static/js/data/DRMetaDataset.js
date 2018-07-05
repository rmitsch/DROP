import Utils from "../Utils.js";

/**
 * Wrapper class providing the specified dataset itself plus the corresponding crossfilter context and various utility
 * methods.
 * Note that this class includes a few custom tweaks regarding which dimensions and groups to generate.
 * This class might be split and only used as abstract base class, once the prototype is extended to other operators -
 * typically, operators have different requirements regarding their datasets and their capabilities. This could be
 * reflected by a diverging DRMetaDataset class hierarchy.
 */
export default class DRMetaDataset
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
    constructor(name, data, metadata, binCount) {
        this._name              = name;
        this._data              = data;
        this._dataIndicesByID   = {};
        this._metadata          = metadata;
        this._binCount          = binCount;
        // Defines how much padding (relative to the shown interval) any axis should have.
        this._axisPaddingRatio  = 6.0;

        // Extract categorical hyperparameter for later shorthand usage.
        this._categoricalHyperparameterSet = this._extractCategoricalHyperparameters();

        // Store ID-to-index references for data elements.
        for (let i = 0; i < this._data.length; i++) {
            this._dataIndicesByID[this._data[i].id] = i;
        }

        let minRnx = 1000;
        for (let i = 0; i < this._data.length; i++) {
            minRnx = this._data[i]["r_nx"] < minRnx ? this._data[i]["r_nx"] : minRnx;
        }
        let rnxvals = [];
        for (let i = 0; i < this._data.length; i++) {
            this._data[i]["r_nx"] = this._data[i]["n_iter"] / 2.5 + Math.random() * 100;
            rnxvals.push(this._data[i]["r_nx"]);
        }

        console.log(rnxvals);


        // Translate categorical variables into numerical ones; store maps for translation.
        this._categoricalToNumericalValues = {};
        this._numericalToCategoricalValues = {};
        this._discretizeCategoricalHyperparameters();

        // Set up containers for crossfilter data.
        this._crossfilter   = crossfilter(this._data);
        this._cf_dimensions = {};
        this._cf_extrema    = {};
        this._cf_groups     = {};
        this._cf_intervals  = {};



        // Set up singular dimensions (one dimension per attribute).
        this._initSingularDimensionsAndGroups();

        // Set up binary dimensions (for scatterplots).
        this.initBinaryDimensionsAndGroups(true);

        // Set up histogram dimensions.
        this._initHistogramDimensionsAndGroups();

        // Create series mapping.
        // Since for the intended use case (i. e. DROP) it is to be expected to need series variant w.r.t. each possible
        // hyperparameter, in makes sense to calculate all of them beforehand.
        this._seriesMappingByHyperparameter = this._generateSeriesMappingForHyperparameters();
    }

    /**
     * Returns dict for translating column headers in JSON/dataframe into human-readable titles.
     * @returns {{n_components: string, perplexity: string, early_exaggeration: string, learning_rate: string, n_iter: string, angle: string, metric: string, r_nx: string, b_nx: string, stress: string, classification_accuracy: string, separability_metric: string, runtime: string}}
     */
    static translateAttributeNames()
    {
        return {
            "n_components": "Dimensions",
            "perplexity": "Perplexity",
            "early_exaggeration": "Early exagg.",
            "learning_rate": "Learning rate",
            "n_iter": "Iterations",
            "angle": "Angle",
            "metric": "Dist. metric",
            "r_nx": "R<sub>nx</sub>",
            "b_nx": "B<sub>nx</sub>",
            "stress": "Stress",
            "classification_accuracy": "Accuracy",
            "separability_metric": "Silhouette",
            "runtime": "Runtime"
        }
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
        this._cf_extrema[attribute].min -= this._cf_intervals[attribute] / this._axisPaddingRatio;
        this._cf_extrema[attribute].max += this._cf_intervals[attribute] / this._axisPaddingRatio;
    }

    /**
     * Calculates extrema for all histogram dimensions/groups.
     * @param attribute
     * @param dataType "categorical" or "numerical". Distinction is necessary due to diverging structure of histogram
     * data.
     */
    _calculateHistogramExtremaForAttribute(attribute, dataType)
    {
        // Calculate extrema for histograms.
        let histogramAttribute  = attribute + "#histogram";
        let sortedData          = this._cf_groups[histogramAttribute].all();

        // Sort data by number of entries in this attribute's histogram.
        sortedData.sort(function(entryA, entryB) {
            let countA = dataType === "numerical" ? entryA.value.count : entryA.value;
            let countB = dataType === "numerical" ? entryB.value.count : entryB.value;

            return countA > countB ? 1 : (countB > countA ? -1 : 0);
        });

        // Determine extrema.
        this._cf_extrema[histogramAttribute] = {
            min: ((dataType === "numerical") ? sortedData[0].value.count : sortedData[0].value),
            max: ((dataType === "numerical") ? sortedData[sortedData.length - 1].value.count : sortedData[sortedData.length - 1].value)
        };

        // Update extrema by padding values (hardcoded to 10%) for x-axis.
        this._cf_intervals[histogramAttribute]   = this._cf_extrema[histogramAttribute].max - this._cf_extrema[histogramAttribute].min;
        this._cf_extrema[histogramAttribute].min -= this._cf_intervals[histogramAttribute] / this._axisPaddingRatio;
        this._cf_extrema[histogramAttribute].max += this._cf_intervals[histogramAttribute] / this._axisPaddingRatio;
    }

    /**
     * Initializes singular dimensions.
     * Note: Creates dimensions for all attribute by default. If not desired, columns have to be dropped beforehand.
     */
    _initSingularDimensionsAndGroups()
    {
        let hyperparameterList = Utils.unfoldHyperparameterObjectList(this._metadata.hyperparameters);

        // -------------------------------------
        // Create dimensions and groups.
        // -------------------------------------

        // Create dimensions for hyperparameters and objectives.
        for (let attribute of hyperparameterList.concat(this._metadata.objectives)) {
            this._initSingularDimension(attribute);

            // If attribute is categorical: Also create dimension for numerical representation of this attribute.
            if (this._categoricalHyperparameterSet.has(attribute)) {
                this._initSingularDimension(attribute + "*");
            }
        }
    }

    /**
     * Initializes singular dimensions and calculates extrema for specified attribute.
     * @param attribute
     * @private
     */
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
     * Initializes singular dimensions w.r.t. histograms.
     */
    _initHistogramDimensionsAndGroups()
    {
        let hyperparameters     = Utils.unfoldHyperparameterObjectList(this._metadata.hyperparameters);
        let attributes          = hyperparameters.concat(this.metadata.objectives);
        let instance            = this;
        let histogramAttribute  = null;

        for (let i = 0; i < attributes.length; i++) {
            let attribute       = attributes[i];
            histogramAttribute  = attribute + "#histogram";


            let binWidth        = instance._cf_intervals[attribute] / this._binCount;

            if (attribute === "r_nx")
                binWidth = (Math.round(instance._cf_intervals[attribute] / this._binCount * 100 ) / 100).toFixed(2);


            for (let j = 0; j < this._data.length; j++) {
                let value   = this._data[j][attribute];
                let extrema = this._cf_extrema[attribute];
                if (value <= extrema[0])
                    value = extrema[0];
                else if (value >= extrema[1])
                    value = extrema[1] - binWidth;

                this._data[j][histogramAttribute] = (Math.round(value / binWidth) * binWidth);
                if (attribute === "r_nx")
                    console.log(this._data[j][histogramAttribute]);
            }


            // for (let j = 0; j < this._data.length; j++) {
            //     let value   = this._data[j][attribute];
            //     let extrema = this._cf_extrema[attribute];
            //     if (value <= extrema[0])
            //         value = extrema[0];
            //     else if (value >= extrema[1])
            //         value = extrema[1] - binWidth;
            //
            //     this._data[j][attribute + "#histogram"] = (Math.round(value / binWidth) * binWidth);
            //     if (attribute === "r_nx")
            //         console.log(this._data[j][attribute + "#histogram"]);
            // }

            // If this is a numerical hyperparameter or an objective: Returned binned width.
            if (i < hyperparameters.length &&
                this._metadata.hyperparameters[i].type === "numeric" ||
                i >= hyperparameters.length
            ) {
                // Dimension with rounded values (for histograms).
                this._cf_dimensions[histogramAttribute] = this._crossfilter.dimension(
                    function (d) { return d[histogramAttribute]; }
                );

                // Create group for histogram.
                this._cf_groups[histogramAttribute] = this._generateGroupWithCounts(
                    histogramAttribute, [histogramAttribute]
                );

                // Calculate extrema.
                this._calculateHistogramExtremaForAttribute(attribute, "numerical");
            }

            // Else if this is a categorical hyperparameter: Return value itself.
            else {
                this._cf_dimensions[histogramAttribute] = this._crossfilter.dimension(
                    function (d) { return d[attribute]; }
                );

                // Create group for histogram.
                this._cf_groups[attribute + "#histogram"] = this._cf_dimensions[attribute + "#histogram"].group().reduceCount();

                // Calculate extrema.
                this._calculateHistogramExtremaForAttribute(attribute, "categorical");
            }

        }
    }

    /**
     * Initializes binary dimensions through cartesian product - one dimension per combination of
     * hyperparameter-objective and objective-objective pairings.
     * @param includeGroups Determines whether groups for binary dimensions should be generated as well.
     */
    initBinaryDimensionsAndGroups(includeGroups = true)
    {
        // Transform list of hyperparameter objects into list of hyperparameter names.
        let hyperparameters             = Utils.unfoldHyperparameterObjectList(this._metadata.hyperparameters);
        let categoricalHyperparameters  = this._extractCategoricalHyperparameters();

        // Hyperparameter-objective and objective-objective pairings.
        for (let attribute1 of hyperparameters.concat(this._metadata.objectives)) {
            // Check if attribute is a categorical hyperparameter.
            // Use suffix "*" if attribute is categorical (and hence its numerical representation is to be used in
            // scatterplots).
            let processedAttribute1         = attribute1 + (categoricalHyperparameters.has(attribute1) ? "*" : "");

            for (let attribute2 of this._metadata.objectives) {
                let combinedKey     = processedAttribute1 + ":" + attribute2;
                let transposedKey   = attribute2 + ":" + processedAttribute1;

                // Only create new dimensions if transposed key didn't appear so far (i. e. the reverse combination
                // didn't already appear -> for A:B check if B:A was already generated).
                // Also: Drop auto-references (A:A).
                if (!(combinedKey in this._cf_dimensions) &&
                    !(transposedKey in this._cf_dimensions) &&
                    attribute1 !== attribute2
                ) {
                    // Create combined dimension (for scatterplot).
                    this._cf_dimensions[combinedKey] = this._crossfilter.dimension(
                        function(d) {
                            return [d[processedAttribute1], d[attribute2]];
                        }
                    );

                    // Mirror dimension to transposed key.
                    this._cf_dimensions[transposedKey] = this._cf_dimensions[combinedKey];

                    // Create group for scatterplot.
                    this._cf_groups[combinedKey] = this._generateGroupWithCounts(
                        combinedKey, [attribute1, attribute2]
                    );

                    // Mirror group to transposed key.
                    this._cf_groups[transposedKey] = this._cf_groups[combinedKey];
                }
            }
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

                if (true || primitiveAttributes.length === 1 && primitiveAttributes[0] === "r_nx") {
                    // console.log("item.id = " + item.id);
                    let match = false;
                    let values = [];
                    for (let i = 0; i < elements.items.length; i++) {
                        //console.log(elements.items[i].id);
                        // console.log("bleb");
                        values.push(elements.items[i]["r_nx"]);

                        // Compare hyperparameter signature.
                        if (item.id === elements.items[i].id) {
                            match = true;
                            elements.items.splice(i, 1);
                            //elements.items.splice(elements.items.indexOf(item), 1);
                            elements.count--;
                            if (primitiveAttributes.length === 1 && primitiveAttributes[0] === "r_nx#histogram")
                                console.log("match = " + match);
                        }
                    }

                    //console.log(values.sort((a, b) => a - b));
                }
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
        // represenation in dataset.
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

    get name()
    {
        return this._name;
    }

    get data()
    {
        return this._data;
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

    get idToSeriesMappingByHyperparameter()
    {
        return this._seriesMappingByHyperparameter;
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
}