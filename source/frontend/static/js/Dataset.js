import Utils from "./Utils.js";

/**
 * Wrapper class providing the specified dataset itself plus the corresponding crossfilter context and various utility
 * methods.
 * Note that this class includes a few custom tweaks regarding which dimensions and groups to generate.
 * This class might be split and only used as abstract base class, once the prototype is extended to other operators -
 * typically, operators have different requirements regarding their datasets and their capabilities. This could be
 * reflected by a diverging Dataset class hierarchy.
 */
export default class Dataset
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

        // Store ID-to-index references for data elements.
        for (let i = 0; i < this._data.length; i++) {
            this._dataIndicesByID[this._data[i].id] = i;
        }

        // Set up containers for crossfilter data.
        this._crossfilter   = crossfilter(this._data);
        this._cf_dimensions = {};
        this._cf_extrema    = {};
        this._cf_groups     = {};
        this._cf_intervals  = {};

        // Set up singular dimensions (one dimension per attribute).
        this.initSingularDimensionsAndGroups();

        // Calculate extrema.
        this.calculateExtrema();

        // Set up histogram dimensions.
        this.initHistogramDimensionsAndGroups();

        // Create series mapping.
        // Since for the intended use case (i. e. DROP) it is to be expected to need series variant w.r.t. each possible
        // hyperparameter, in makes sense to calculate all of them beforehand.
        this._seriesMappingByHyperparameter = this._generateSeriesMappingForHyperparameters();
    }

    /**
     * Calculates extrema for all singular dimensions.
     */
    calculateExtrema()
    {
        let hyperparameters = [];
        for (let hyperparam in this._metadata.hyperparameters) {
            hyperparameters.push(this._metadata.hyperparameters[hyperparam].name);
        }

        for (let attribute of hyperparameters.concat(this._metadata.objectives)) {
            this._cf_extrema[attribute] = {
                min: this._cf_dimensions[attribute].bottom(1)[0][attribute],
                max: this._cf_dimensions[attribute].top(1)[0][attribute]
            };

            // Update extrema by padding values (hardcoded to 10%) for x-axis.
            this._cf_intervals[attribute]   = this._cf_extrema[attribute].max - this._cf_extrema[attribute].min;
            this._cf_extrema[attribute].min -= this._cf_intervals[attribute] / 3.0;
            this._cf_extrema[attribute].max += this._cf_intervals[attribute] / 3.0;
        }
    }

    /**
     * Initializes singular dimensions.
     * Note: Creates dimensions for all attribute by default. If not desired, columns have to be dropped beforehand.
     */
    initSingularDimensionsAndGroups()
    {
        let hyperparameters = Utils.unfoldHyperparameterObjectList(this._metadata.hyperparameters);

        // -------------------------------------
        // Create dimensions and grouops.
        // -------------------------------------

        // Create dimensions for hyperparameters and objectives.
        for (let attribute of hyperparameters.concat(this._metadata.objectives)) {
            // Dimension with exact values.
            this._cf_dimensions[attribute] = this._crossfilter.dimension(
                function(d) { return d[attribute]; }
            );
        }
    }

    /**
     * Initializes singular dimensions w.r.t. histograms.
     */
    initHistogramDimensionsAndGroups()
    {
        let hyperparameters = Utils.unfoldHyperparameterObjectList(this._metadata.hyperparameters);
        let attributes = hyperparameters.concat(this.metadata.objectives);
        let instance        = this;

        for (let i = 0; i < attributes.length; i++) {
            let attribute   = attributes[i];

            // If this is a numerical hyperparameter or an objective: Returned binned width.
            if (i < hyperparameters.length &&
                this._metadata.hyperparameters[i].type === "numeric" ||
                i >= hyperparameters.length) {
                // Dimension with rounded values (for histograms).
                this._cf_dimensions[attribute + "#histogram"] = this._crossfilter.dimension(
                    function (d) {
                        let binWidth = instance._cf_intervals[attribute] / instance._binCount;
                        return (Math.round(d[attribute] / binWidth) * binWidth);
                    }
                );

                // Create group for histogram.
                this._cf_groups[attribute + "#histogram"] = this._generateGroupForHistogram(attribute);
            }

            // Else if this is a categorical hyperparameter: Return value itself.
            else {
                this._cf_dimensions[attribute + "#histogram"] = this._crossfilter.dimension(
                    function (d) { return d[attribute]; }
                );

                // Create group for histogram.
                this._cf_groups[attribute + "#histogram"] = this._cf_dimensions[attribute + "#histogram"].group().reduceCount();
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
        let hyperparameters = [];
        for (let hyperparam in this._metadata.hyperparameters) {
            hyperparameters.push(this._metadata.hyperparameters[hyperparam].name);
        }

        // Hyperparameter-objective and objective-objective pairings.
        for (let attribute1 of hyperparameters.concat(this._metadata.objectives)) {
            for (let attribute2 of this._metadata.objectives) {
                let combinedKey     = attribute1 + ":" + attribute2;
                let transposedKey   = attribute2 + ":" + attribute1;

                // Only create new dimensions if transposed key didn't appear so far (i. e. the reverse combination
                // didn't already appear -> for A:B check if B:A was already generated).
                // Also: Drop auto-references (A:A).
                if (
                    !(combinedKey in this._cf_dimensions) &&
                    !(transposedKey in this._cf_dimensions) &&
                    attribute1 !== attribute2
                ) {
                    // Create combined dimension (for scatterplot).
                    this._cf_dimensions[combinedKey] = this._crossfilter.dimension(
                        function(d) { return [d[attribute1], d[attribute2]]; }
                    );
                    // Mirror dimension to transposed key.
                    this._cf_dimensions[transposedKey] = this._cf_dimensions[combinedKey];

                    // Create group for scatterplot.
                    this._cf_groups[combinedKey] = this._cf_dimensions[combinedKey].group().reduce(
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
                    );
                    // Mirror group to transposed key.
                    this._cf_groups[transposedKey] = this._cf_groups[combinedKey];
                }
            }
        }
    }

    /**
     * Generates crossfilter group for histogram.
     * @param attribute
     * @returns Newly generated group.
     * @private
     */
    _generateGroupForHistogram(attribute)
    {
        return this._cf_dimensions[attribute + "#histogram"].group().reduce(
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