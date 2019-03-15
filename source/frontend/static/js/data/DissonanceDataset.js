import Utils from "../Utils.js";
import Dataset from "./Dataset.js";

/**
 * Class holding raw and processed data for sample-in-model dissonance data.
 */
export default class DissonanceDataset extends Dataset
{
    /**
     * Constructs new DissonanceDataset.
     * @param name
     * @param data
     * @param binCounts
     * @param drModelMetadata Instance of DRMetaDataset containing metadata of DR models.
     * @param supportedDRModelMeasure DR model measure(s) to consider in data.
     */
    constructor(name, data, binCounts, drModelMetadata, supportedDRModelMeasure)
    {
        super(name, data);

        this._axisPaddingRatio          = 0;
        this._binCounts                 = binCounts;
        this._binWidths                 = {};
        this._drModelMetadata           = drModelMetadata;
        this._supportedDRModelMeasure   = supportedDRModelMeasure;

        // Initialize extrema. Note: Assumes our metrics are [0, 1]-normalized.
        this._colExtrema      = {min: 0, max: 1};
        this._rowExtrema      = {min: 0, max: 1};

        // Store sort settings.
        this._sortSettings              =  this._initializeSortingSettings();
        // Define supported sorting criterions.
        this._allowedCriterions         = ["natural", "asc", "desc"];

        // Add DR model measure to records.
        this._complementModelMeasures();
        // Set up containers for crossfilter data.
        this._crossfilter = crossfilter(this._data);

        // Initialize crossfilter data.
        this._initSingularDimensionsAndGroups();
        this._initHistogramDimensionsAndGroups();
        this._initBinaryDimensionsAndGroups();
    }

    /**
     * Initialize default sort settings.
     */
    _initializeSortingSettings()
    {
        let sortSettings = {
            horizontal: {
                criterion: "natural",
                recordValueToNaturalBinIndex: {},
                sortOrderToNaturalOrder: {},
                naturalOrderToSortOrder: {}
            },
            vertical: {
                criterion: "natural",
                recordValueToNaturalBinIndex: {},
                sortOrderToNaturalOrder: {},
                naturalOrderToSortOrder: {}
            },
            heatmap: {
                recordValueToNaturalBinIndex: {},
                sortOrderToNaturalOrder: {},
                naturalOrderToSortOrder: {},
                selectedNaturalCells: [],
                selectedOrderedCells: []
            }
        };

        // Set up initial values for order translations.
        for (let i = 0; i < this._binCounts.x; i++) {
           sortSettings.horizontal.naturalOrderToSortOrder[i] = i;
           sortSettings.horizontal.sortOrderToNaturalOrder[i] = i;
        }
        for (let i = 0; i < this._binCounts.y; i++) {
           sortSettings.vertical.naturalOrderToSortOrder[i] = i;
           sortSettings.vertical.sortOrderToNaturalOrder[i] = i;
        }

        return sortSettings;
    }
    /**
     * Complements records with the corresponding DR model's measure.
     * @private
     */
    _complementModelMeasures()
    {
        let drModelIDToMeasureValue = {};

        // ----------------------------------------------------
        // 1. Add DR model measures to samples-in-models.
        // ----------------------------------------------------

        // Iterate over DR model indices, store corresponding measure.
        for (let drModelID in this._drModelMetadata._dataIndicesByID) {
            let drModelIndex                    = this._drModelMetadata._dataIndicesByID[drModelID];
            drModelIDToMeasureValue[drModelID]  = this._drModelMetadata._data[drModelIndex][this._supportedDRModelMeasure];
        }

        // Update internal dataset with model measure for samples-in-models.
        for (let sampleInModelIndex in this._data) {
            this._data[sampleInModelIndex][this._supportedDRModelMeasure] = drModelIDToMeasureValue[
                this._data[sampleInModelIndex].model_id
            ];
        }
    }

    _initSingularDimensionsAndGroups()
    {
        let attributes = ["sample_id", "model_id", "measure", this._supportedDRModelMeasure];

        for (let attribute of attributes) {
            this._cf_dimensions[attribute] = this._crossfilter.dimension(
                function(d) { return d[attribute]; }
            );

            // Find extrema.
            this._calculateSingularExtremaByAttribute(attribute);
        }
    }

    _initBinaryDimensionsAndGroups()
    {
        let scope           = this;
        let attribute       = "samplesInModelsMeasure:sampleDRModelMeasure";
        let colAttribute    = "measure";
        let rowAttribute    = this._supportedDRModelMeasure;
        // Extrema can be set to 0/1, since we expect both measures to be between 0 and 1.
        // Note that this holds iff measure is explicitly standardized to 0 <= x <= 1.
        this._colExtrema    = {min: 0, max: 1}; // this._cf_extrema[colAttribute];
        this._rowExtrema    = {min: 0, max: 1}; // this._cf_extrema[rowAttribute];

        // Workaround: min has to be 0 - probably error in backend calculation.
        this._rowExtrema.min = 0;
        let rowBinWidth     = (this._rowExtrema.max - this._rowExtrema.min) / this._binCounts.y;
        let colBinWidth     = (this._colExtrema.max - this._colExtrema.min) / this._binCounts.x;

        // 1. Create dimension for sample-in-model measure vs. sample's DR model measure.
        this._cf_dimensions[attribute] = this._crossfilter.dimension(
            // Determine bin number and return as value (i. e. we represent the bin number instead
            // of the real, rounded value).
            function(d) {
                // (a) Get row number.
                let rowValue = d[rowAttribute];
                if (rowValue <= scope._rowExtrema.min)
                    rowValue = scope._rowExtrema.min;
                else if (rowValue >= scope._rowExtrema.max) {
                    rowValue = scope._rowExtrema.max - rowBinWidth;
                }

                // (b) Get column number.
                let colValue = d[colAttribute];
                if (colValue <= scope._colExtrema.min)
                    colValue = scope._colExtrema.min;
                else if (colValue >= scope._colExtrema.max) {
                    colValue = scope._colExtrema.max - colBinWidth;
                }

                // Calculate bin index and use as column/row number.
                return [+Math.floor(colValue / colBinWidth), +Math.floor(rowValue / rowBinWidth)];
            }
        );

        // 2. Define group as sum of intersection sample_id/model_id - since only one element per
        // group exists, sum works just fine.
        this._cf_groups[attribute] = this._cf_dimensions[attribute]
            .group()
            .reduce(
                function(elements, item) {
                    if (item.model_id !== -1) {
                        elements.count++;
                        if (!(item.model_id in elements.ids))
                            elements.ids[item.model_id] = 0;
                        elements.ids[item.model_id]++;
                    }
                    return elements;
                },
                function(elements, item) {
                    if (item.model_id !== -1) {
                        elements.count--;
                        elements.ids[item.model_id]--;
                    }
                    return elements;
                },
                function() {
                    return { count: 0, ids: {} };
                }
            );

        // 3. Find extrema.
        this._calculateHistogramExtremaForAttribute(attribute);

        // 4. Implement sorting mechanism in dc.js/crossfilter.js framework.
        // this._initHeatmapSortingMechanism(attribute);
    }

    /**
     * Initializes singular dimensions w.r.t. histograms.
     */
    _initHistogramDimensionsAndGroups()
    {
        let scope                           = this;
        let yAttribute                      = "measure";
        let extrema                         = {min: 0, max: 1};
        let histogramAttribute              = "samplesInModels#" + yAttribute;
        let binWidth                        = null;
        let sortSettings                    = null;

        // -----------------------------------------------------
        // 1. Create group for histogram on x-axis (SIMs).
        // -----------------------------------------------------

        this._binWidths[histogramAttribute] = (extrema.max - extrema.min) / this._binCounts.x;
        binWidth                            = this._binWidths[histogramAttribute];
        sortSettings                        = scope._sortSettings.horizontal;

        // Form group.
        this._cf_groups[histogramAttribute] = this._cf_dimensions[yAttribute]
            .group(function(value) {
                let originalValue = value;
                if (value <= extrema.min)
                    value = extrema.min;
                else if (value >= extrema.max) {
                    value = extrema.max - binWidth;
                }

                // Deduct minimum (in case our range does not start with zero).
                value -= extrema.min;

                // Store association of record value for this dimension to bin index.
                sortSettings.recordValueToNaturalBinIndex[originalValue] = Math.floor(value / binWidth);
                return sortSettings.recordValueToNaturalBinIndex[originalValue];
            })
            // Custome reducer ignoring dummys with model_id === -1.
            .reduce(
                function(elements, item) {
                    if (item.model_id !== -1) {
                        elements.count++;
                        if (!(item.model_id in elements.ids))
                            elements.ids[item.model_id] = 0;
                        elements.ids[item.model_id]++;
                    }
                    return elements;
                },
                function(elements, item) {
                    if (item.model_id !== -1) {
                        elements.count--;
                        elements.ids[item.model_id]--;
                    }
                    return elements;
                },
                function() {
                    return { count: 0, ids: {} };
                }
            );

        // Calculate extrema.
        this._calculateHistogramExtremaForAttribute(histogramAttribute);

        // Implement sorting mechanism in dc.js/crossfilter.js framework.
        this._initHistogramSortingMechanism(yAttribute, this._sortSettings.horizontal);

        // -----------------------------------------------------
        // 2. Create group for histogram on y-axis (number of
        //    samples-in-models with given DR modelmeasure).
        // -----------------------------------------------------

        yAttribute                          = this._supportedDRModelMeasure;
        histogramAttribute                  = "samplesInModels#" + yAttribute;
        this._binWidths[histogramAttribute] = (extrema.max - extrema.min) / this._binCounts.y;
        binWidth                            = this._binWidths[histogramAttribute];
        sortSettings                        = scope._sortSettings.vertical;

        // Form group.
        this._cf_groups[histogramAttribute] = this._cf_dimensions[yAttribute]
            .group(function(value) {
                let originalValue = value;
                if (value <= extrema.min)
                    value = extrema.min;
                else if (value >= extrema.max) {
                    value = extrema.max - binWidth;
                }

                // Deduct minimum (in case our range does not start with zero).
                value -= extrema.min;

                // Store association of record value for this dimension to bin index.
                sortSettings.recordValueToNaturalBinIndex[originalValue] = Math.floor(value / binWidth);
                return sortSettings.recordValueToNaturalBinIndex[originalValue];
            })
            // Custome reducer ignoring dummys with model_id === -1.
            .reduce(
                function(elements, item) {
                    if (item.model_id !== -1) {
                        elements.count++;
                        if (!(item.model_id in elements.ids))
                            elements.ids[item.model_id] = 0;
                        elements.ids[item.model_id]++;
                    }
                    return elements;
                },
                function(elements, item) {
                    if (item.model_id !== -1) {
                        elements.count--;
                        elements.ids[item.model_id]--;
                    }
                    return elements;
                },
                function() {
                    return { count: 0, ids: {} };
                }
            );

        // Calculate extrema.
        this._calculateHistogramExtremaForAttribute(histogramAttribute);

        // Implement sorting mechanism in dc.js/crossfilter.js framework.
        this._initHistogramSortingMechanism(yAttribute, this._sortSettings.vertical);
    }

    /**
     * Initializes sorting for dissonance histogram data.
     * Has to use a few workarounds; hencesome custom code.
     * @param attribute
     * @param settings Settings object to use.
     * @private
     */
    _initHistogramSortingMechanism(attribute, settings)
    {
        let scope = this;

        this._cf_dimensions[attribute + "#sort"] = {
            // For reset: filter(null).
            filter: function(f) {
                if (f !== null)
                    throw new Error("DissonanceDataset._initHistogramSortingMechanism: filter(f) with f !== null called.");
                scope._cf_dimensions[attribute].filter(null);
            },

            // Filter selected range.
            filterRange: function(r) {
                // 1. Get all bins corresponding with selected range.
                //    In order to do that, we pick the corresponding natural filtered bins
                //    and compare against them.
                let filteredNaturalBinIndices = [];
                for (let i = Math.floor(r[0]); i < Math.ceil(r[1]); i++) {
                    filteredNaturalBinIndices.push(settings.sortOrderToNaturalOrder[i]);
                }

                // 2. Set filtering function for individual records.
                scope._cf_dimensions[attribute].filterFunction(function(value) {
                    // Check if this record's natural bin was selected.
                    return filteredNaturalBinIndices.indexOf(
                        settings.recordValueToNaturalBinIndex[value]
                    ) > -1;
                });
            }
        };
    }

    /**
     * Initializes sorting for dissonance heatmap data.
     * Has to use a few workarounds; hence some custom code.
     * @param attribute
     * @param settings Settings object to use.
     * @private
     */
    _initHeatmapSortingMechanism(attribute, settings)
    {
        let scope = this;

        this._cf_dimensions[attribute + "#sort"] = {
            // For reset: filter(null).
            filter: function(f) {
                if (f !== null)
                    throw new Error("DissonanceDataset._initHistogramSortingMechanism: filter(f) with f !== null called.");

                // Reset filters.
                scope._sortSettings.heatmap.selectedOrderedCells = [];
                scope._sortSettings.heatmap.selectedNaturalCells = [];
                scope._cf_dimensions[attribute].filter(null);
            },

            filterFunction: function(f) {
                // Set filtering function for individual records.
                scope._cf_dimensions[attribute].filterFunction(function(value) {
                    return scope._sortSettings.heatmap.selectedNaturalCells.find(
                        cellCoords => cellCoords[0] === value[0] && cellCoords[1] === value[1]
                    ) !== undefined;
                });
            }
        };
    }

    /**
     * Fills up gaps in collection of bins.
     * @param originalBins
     * @param interval
     * @returns {any}
     * @private
     */
    _fillGapsInHistogramBins(originalBins, interval)
    {
        let bins        = JSON.parse(JSON.stringify(originalBins));
        let gapIndices  = new Set();

        for (let i = interval.min; i < interval.max; i++)
            gapIndices.add(i);
        for (let bin of bins)
            gapIndices.delete(bin.key);
        for (let gapIndex of gapIndices)
            bins.push({key: gapIndex, value: {count: 0, ids: {}}});

        return bins;
    }

    /**
     * Sorts histogram group by specified criterion (e. g. "asc", "desc" or "natural").
     * @param group
     * @param settings Corresponding settings object.
     * @param sortCriterion
     * @param interval
     * @returns {{all: all}}
     * @private
     */
    sortHistogramGroup(group, settings, sortCriterion, interval)
    {
        // ----------------------------------------------
        // Sort specified groups and update internal
        // information.
        // ----------------------------------------------

        let scope = this;

        return {
            all: function() {
                let unsortedData                    = scope._fillGapsInHistogramBins(group.all(), interval);
                let sortedData                      = JSON.parse(JSON.stringify(unsortedData));

                settings.sortOrderToNaturalOrder    = {};
                settings.naturalOrderToSortOrder    = {};
                settings.criterion                  = sortCriterion;

                // Sort data by number of entries in this attribute's histogram.
                sortedData.sort(function(entryA, entryB) {
                    let countA = typeof entryA.value === "number" ? +entryA.value : +entryA.value.count;
                    let countB = typeof entryB.value === "number" ? +entryB.value : +entryB.value.count;

                    switch (sortCriterion) {
                        case "natural":
                            return +entryA.key > +entryB.key ? 1 : (+entryB.key > +entryA.key ? -1 : 0);

                        case "asc":
                            return countA > +countB ? 1 : (countB > countA ? -1 : 0);

                        case "desc":
                            return countA < +countB ? 1 : (countB < countA ? -1 : 0);

                        default:
                            throw new RangeError("Sorting criterion " + sortCriterion + " not supported.");
                    }
                });

                // Store association between natural and ordered bin index.
                for (let i = 0; i < sortedData.length; i++) {
                    settings.sortOrderToNaturalOrder[unsortedData[i].key]   = sortedData[i].key;
                    settings.naturalOrderToSortOrder[sortedData[i].key]     = unsortedData[i].key;
                    sortedData[i].key                                       = unsortedData[i].key;
                }

                return sortedData;
            }
        };

    }

    /**
     * Sorts heatmap/binary group by specified criterion (e. g. "asc", "desc" or "natural").
     * @param group
     * @param settings Corresponding settings object.
     * @returns {{all: all}}
     * @private
     */
    sortHeatmapGroup(group, settings)
    {
        // ----------------------------------------------
        // Sort specified groups and update internal
        // information.
        // ----------------------------------------------

        let scope = this;

        return {
            all: function() {
                let sortSettings                    = {
                    x: scope._sortSettings.horizontal,
                    y: scope._sortSettings.vertical
                };
                let sortedData                      = JSON.parse(JSON.stringify(group.all()));
                settings.sortOrderToNaturalOrder    = {};
                settings.naturalOrderToSortOrder    = {};

                // Swap rows in accordance with defined sort order.
                for (let i = 0; i < sortedData.length; i++) {
                    sortedData[i].key[0] = sortSettings.x.naturalOrderToSortOrder[sortedData[i].key[0]];
                    sortedData[i].key[1] = sortSettings.y.naturalOrderToSortOrder[sortedData[i].key[1]];
                }

                return sortedData;
            }
        };
    }

    /**
     * Calculates extrema for all histogram dimensions/groups.
     * See https://stackoverflow.com/questions/25204782/sorting-ordering-the-bars-in-a-bar-chart-by-the-bar-values-with-dc-js.
     * @param histogramAttribute
     */
    _calculateHistogramExtremaForAttribute(histogramAttribute)
    {
        // Calculate extrema for histograms.
        let sortedData = JSON.parse(JSON.stringify(this._cf_groups[histogramAttribute].all()))

        // Sort data by number of entries in this attribute's histogram.
        sortedData.sort(function(entryA, entryB) {
            let countA = entryA.value.count;
            let countB = entryB.value.count;

            return countA > countB ? 1 : (countB > countA ? -1 : 0);
        });

        // Determine extrema.
        this._cf_extrema[histogramAttribute] = {
            min: (sortedData[0].value.count),
            max: (sortedData[sortedData.length - 1].value.count)
        };

        // Update extrema by padding values (hardcoded to 10%) for x-axis.
        this._cf_intervals[histogramAttribute] =
            this._cf_extrema[histogramAttribute].max -
            this._cf_extrema[histogramAttribute].min;
    }
}