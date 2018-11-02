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
        this._binCount          = binCount;
        this._binCountSSP       = 10;
        // Maps for translation of categorical variables into numerical ones.
        this._categoricalToNumericalValues = {};
        this._numericalToCategoricalValues = {};

        // Extract categorical hyperparameter for later shorthand usage.
        this._categoricalHyperparameterSet = this._extractCategoricalHyperparameters();

        // Update record metadata before further preprocessing.
        this._updateRecordMetadata();

        // todo Bin data for scatterplots, base all dimensions and groups on binned dataset.
        this._crossfilterData = {};
        this._binDataForSSPs();

        // Set up containers for crossfilter data.
        this._crossfilter = crossfilter(this._data);

        // Set up singular dimensions (one dimension per attribute).
        this._initSingularDimensionsAndGroups();

        // Set up binary dimensions (for scatterplots).
        this._initBinaryDimensionsAndGroups(true);

        // Set up histogram dimensions.
        this._initHistogramDimensionsAndGroups();

        // Create series mapping.
        // Since for the intended use case (i. e. DROP) it is to be expected to need series variant w.r.t. each possible
        // hyperparameter, in makes sense to calculate all of them beforehand.
        this._seriesMappingByHyperparameter = this._generateSeriesMappingForHyperparameters();
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
     * Bin data by metric & objective values for scattered scree plots.
     * @private
     */
    _binDataForSSPs()
    {
        // ---------------------------------------------------
        // 1. Find extrema for each objective.
        // ---------------------------------------------------

        let extrema = {};
        let attributes  = Utils.unfoldHyperparameterObjectList(this._metadata.hyperparameters).concat(this._metadata.objectives);

        for (let attribute of attributes) {
            extrema[attribute] = {max: -Infinity, min: Infinity};
        }

        for (let record of this._data) {
            for (let attr of attributes) {
                extrema[attr].min = extrema[attr].min <= record[attr] ? extrema[attr].min : record[attr];
                extrema[attr].max = extrema[attr].max >= record[attr] ? extrema[attr].max : record[attr];
            }
        }

        // ---------------------------------------------------
        // 2. For each record: Match up with corresponding
        // bin for all attributes.
        // ---------------------------------------------------

        let unaryBinning = this._constructUnaryBinning(this._determineBins(extrema));

        // ---------------------------------------------------
        // 3. For all SSP attribute (hyp/obj, obj/obj)
        // combinations, build structure containing
        // records whose values correspond to these inter-
        // secting sets.
        // ---------------------------------------------------

        // group for each hyp/obj, obj/obj combination.
        let binaryBinning = this._constructBinaryBinning(unaryBinning);

        // ---------------------------------------------------
        // 4. Create crossfilter.js dimensions and groups
        // from binarily binned groups.
        // ---------------------------------------------------

        this._constructDimensionsAndGroups(unaryBinning, binaryBinning);

        // todo
        //  * Use new datasets for plotting in charts.
        //  * Sketch intra-panel B+L.
        //  * Introduce central ID repository for filtering.
        //  * Sketch intra-operator B+L.
        //  * Sketch inter-operator B+L.

        // console.log("finished binning");
    }

    _constructDimensionsAndGroups(unaryBinning, binaryBinning)
    {
        for (let attrCombination of Object.keys(binaryBinning)) {
            let splitAttributes = attrCombination.split(":");

            // Initialize unary constellations, if not defined yet.
            for (let attribute of splitAttributes) {
                if (!(attribute in this._crossfilterData)) {
                    let cf = crossfilter(unaryBinning[attribute]);

                    // Initialize data for attribute.
                    this._crossfilterData[attribute] = {
                        crossfilter: cf,
                        dimensions: {
                            x: this._initSingularDimension_forCrossfilter(cf)
                        },
                        groups: {

                            //             this._generateGroupWithCounts(
                            //     histogramAttribute, [histogramAttribute]
                            // );

                            // Calculate extrema.
                            // this._calculateExtremaForAttribute(attribute, "#histogram", "numerical");
                            //         }
                        }
                    }
                }
            }

            // Initialize binary constellations.
            let cf = crossfilter(binaryBinning[attrCombination]);
            this._crossfilterData[attrCombination] = {
                crossfilter: cf,
                dimensions: {
                    xy: cf.dimension(
                        function (d) {
                            return [d.x, d.y];
                        }
                    )
                },
                groups: {
                // this._generateGroupWithCounts(
                //        combinedKey, [attribute1, attribute2]
                //    );
                }
            };
        }

        // console.log(this._crossfilterData)
        // console.log(palim)
    }

    /**
     * Associates records' IDs with attribute bins they correspond with, thus creating an unary binning of records
     * w.r.t. their values along each dimension.
     * Note: Manipulates groupedRecords.
     * @param groupedRecords
     * @private
     */
    _constructUnaryBinning(groupedRecords)
    {
        let numericAttributes       = JSON.parse(JSON.stringify(this._metadata.objectives));
        let categoricalAttributes   = [];
        let unaryBinning            = groupedRecords;

        for (let hp of this._metadata.hyperparameters) {
            if (hp.type === "numeric")
                numericAttributes.push(hp.name);
            else
                categoricalAttributes.push(hp.name);
        }

        for (let record of this._data) {
            // Associate numerical attributes.
            for (let attr of numericAttributes) {
                for (let i = 0; i < unaryBinning[attr].length; i++) {
                    if (
                        (
                            record[attr] >= unaryBinning[attr][i].lowerThreshold &&
                            record[attr] < unaryBinning[attr][i].upperThreshold
                        ) ||
                        // Allow elements with max. value in last bin.
                        (
                            record[attr] > unaryBinning[attr][i].lowerThreshold &&
                            i === unaryBinning[attr].length - 1
                        )
                    ) {
                        unaryBinning[attr][i].ids.add(record.id);
                        break;
                    }
                }

            }

            // Associate categorical attributes.
            for (let attr of categoricalAttributes) {
                for (let i = 0; i < unaryBinning[attr].length; i++) {
                    if (record[attr] === unaryBinning[attr][i].value) {
                        unaryBinning[attr][i].ids.add(record.id);
                        break;
                    }
                }
            }
        }

        return unaryBinning;
    }

    /**
     * Constructs binary binning for each possible combination of (attr/obj), (obj/obj) values.
     * @param unaryBinning
     * @returns {{}}
     * @private
     */
    _constructBinaryBinning(unaryBinning)
    {
        let attributes = Utils.unfoldHyperparameterObjectList(this._metadata.hyperparameters).concat(this._metadata.objectives);
        let binaryBinning = {};

        for (let attr of attributes) {
            for (let obj of this._metadata.objectives) {
                if (attr !== obj) {
                    binaryBinning[attr + ":" + obj] = [];

                    // Combine all possible values of current attribute + obj.
                    for (let i = 0; i < unaryBinning[attr].length; i++) {
                        for (let j = 0; j < unaryBinning[obj].length; j++) {
                            binaryBinning[attr + ":" + obj].push({
                                x: unaryBinning[attr][i].value,
                                y: unaryBinning[obj][j].value,
                                ids: new Set([...unaryBinning[attr][i].ids].filter(
                                    x => unaryBinning[obj][j].ids.has(x))
                                )
                            });
                        }
                    }
                }
            }
        }

        return binaryBinning;
    }

    /**
     * Determines bins used for bins.
     * Note: Modifies extrema by flooring them.
     * @param extrema
     * @returns {{}} Rounded bin values for each attribute.
     * @private
     */
    _determineBins(extrema)
    {
        let binData  = {};

        // Apply to objectives.
        for (let obj of this._metadata.objectives) {
            binData[obj] = [];
            extrema[obj].binInterval = (extrema[obj].max - extrema[obj].min) / (this._binCountSSP);

            if (extrema[obj].binInterval > 0) {
                for (let i = 0; i < this._binCountSSP; i++) {
                    let lowerThreshold = extrema[obj].min + i * extrema[obj].binInterval;
                    binData[obj].push({
                        lowerThreshold: lowerThreshold,
                        upperThreshold: extrema[obj].min + (i + 1) * extrema[obj].binInterval,
                        value: lowerThreshold,
                        ids: new Set()
                    });
                }
            }
            // Workaround for when only one value available.
            else {
                binData[obj].push({
                    value: extrema[obj].min,
                    ids: new Set()
                });
            }
        }

        // Apply to attributes.
        for (let hyperparam of this._metadata.hyperparameters) {
            let hpName                      = hyperparam.name;
            binData[hpName]  = [];

            if (hyperparam.type === "numeric") {
                let sortedHPValues = hyperparam.values.sort((a, b) => a - b);
                extrema[hpName].binInterval = (extrema[hpName].max - extrema[hpName].min) / (sortedHPValues.length);

                for (let i = 0; i < sortedHPValues.length; i++) {
                    binData[hpName].push({
                        lowerThreshold: sortedHPValues[i],
                        upperThreshold: sortedHPValues[i] + extrema[hpName].binInterval,
                        value: sortedHPValues[i],
                        ids: new Set()
                    });
                }
            }

            else {
                let sortedHPValues = hyperparam.values.sort();

                for (let i = 0; i < hyperparam.values.length; i++) {
                    binData[hpName].push({
                        value: this._categoricalToNumericalValues[hpName][sortedHPValues[i]],
                        stringValue: sortedHPValues[i],
                        ids: new Set()
                    });
                }
            }
        }

        return binData;
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
            "n_neighbors": "Neighbors",
            "min_dist": "Min. Distance",
            "local_connectivity": "Local Conn.",
            "n_epochs": "Iterations",
            // From here: Objectives.
            "r_nx": useHTMLFormatting ? "R<sub>nx</sub>" : "R_nx",
            "b_nx": useHTMLFormatting ? "B<sub>nx</sub>" : "B_nx",
            "stress": "Stress",
            "classification_accuracy": "Accuracy",
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
        let sortedData          = JSON.parse(JSON.stringify(this._cf_groups[modifiedAttribute].all()))

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
     * Initializes singular dimensions for given attribute and crossfilter.
     * @param crossfilter
     * @private
     */
    _initSingularDimension_forCrossfilter(crossfilter)
    {
        // Dimension with exact values.
        let dim = crossfilter.dimension(
            function(d) { return d.value; }
        );

        // Calculate extrema.
        let res = this._calculateSingularExtremaByAttribute_forCrossfilter(dim, crossfilter);

        return { dimension: dim, extrema: res.extrema, interval: res.interval }
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
            let extrema         = this._cf_extrema[attribute];

            // Bin data for current attribute (i. e. hyperparameter or objective).
            for (let j = 0; j < this._data.length; j++) {
                let value   = this._data[j][attribute];
                if (value <= extrema.min)
                    value = extrema.min;
                else if (value >= extrema.max)
                    value = extrema.max - binWidth;

                // Adjust for extrema.
                let binnedValue = binWidth !== 0 ? Math.round((value - extrema.min) / binWidth) * binWidth : 0;
                binnedValue += extrema.min;
                if (binnedValue >= extrema.max)
                    binnedValue = extrema.max - binWidth;

                this._data[j][histogramAttribute] = binnedValue;
            }

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
                this._calculateExtremaForAttribute(attribute, "#histogram", "numerical");
            }

            // Else if this is a categorical hyperparameter: Return value itself.
            else {
                this._cf_dimensions[histogramAttribute] = this._crossfilter.dimension(
                    function (d) { return d[attribute]; }
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
                    // Create combined dimension (for scatterplot)..
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
     * WIP - generateGroupWithCounts() for aggregated dataset.
     * @param dimension
     * @param primitiveAttributes
     * @private
     */
    _generateGroupWithCounts_forAggregated(dimension, primitiveAttributes)
    {
        return this._cf_dimensions[attribute].group().reduce(
            function(elements, item) {
               elements.items.push(item);
               elements.count++;

               return elements;
            },
            function(elements, item) {
                let match = false;

                for (let i = 0; i < elements.items.length && !match; i++) {
                    // Compare hyperparameter signature.
                    if (item.id === elements.items[i].id) {
                        match = true;
                        elements.items.splice(i, 1);
                        elements.count--;
                    }
                }

                return elements;
            },
            function() {
                return {items: [], count: 0};
            }
        );
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
                elements.items.add(item);
                // elements.items.push(item);
                elements.count++;

                return elements;
            },
            function(elements, item) {
                // elements.items.splice(elements.items.indexOf(item), 1);
                elements.items.delete(item);
                elements.count--;
                return elements;
            },
            function() {
                return { items: new Set(), count: 0 };
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

    /**
     * Restores object from instance string.
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
}