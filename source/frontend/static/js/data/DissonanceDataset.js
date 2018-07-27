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

        // Add DR model measure to records.
        this._complementModelMeasures();
        // Set up containers for crossfilter data.
        this._crossfilter = crossfilter(this._data);

        // Initialize crossfilter data.
        this._initSingularDimensionsAndGroups();
        this._initializeBins();
        this._initHistogramDimensionsAndGroups();
        this._initBinaryDimensionsAndGroups();
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
    /**
     * Adds one dummy record per heatmap bin to make sure heatmap doesn't omit any rows/columns.
     * @private
     */
    _initializeBins()
    {
        // ----------------------------------------------------
        // Add n * m dummy records to make sure all bins exist.
        // ----------------------------------------------------

        let rowAttribute    = this._supportedDRModelMeasure;
        let colAttribute    = "measure";
        let rowExtrema      = this._cf_extrema[rowAttribute];
        let colExtrema      = this._cf_extrema[colAttribute];
        let rowBinWidth     = (rowExtrema.max - rowExtrema.min) / this._binCounts.y;
        let colBinWidth     = (colExtrema.max - colExtrema.min) / this._binCounts.x;

        // Generate dummy records for heatmap and add to crossfilter's dataset.
        let dummyRecords = Array(this._binCounts.y * this._binCounts.x).fill(0);
        for (let i = 0; i < this._binCounts.y; i++) {
            for (let j = 0; j < this._binCounts.x; j++) {
                let dummyRecord             = {model_id: -1, sample_id: -1};
                dummyRecord[colAttribute]   = j * colBinWidth;
                dummyRecord[rowAttribute]   = i * rowBinWidth;

                // Add record to crossfilter dataset.
                dummyRecords[i * this._binCounts.x + j] = dummyRecord;
            }
        }
        this._crossfilter.add(dummyRecords);
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
        let attribute       = "samplesInModelsMeasure:sampleDRModelMeasure";
        let colAttribute    = "measure";
        let rowAttribute    = this._supportedDRModelMeasure;
        // Extrema can be set to 0/1, since we expect both measures to be between 0 and 1.
        // Note that this holds iff measure is explicitly standardized to 0 <= x <= 1.
        let colExtrema      = this._cf_extrema[colAttribute];
        let rowExtrema      = this._cf_extrema[rowAttribute];
        // Workaround: min has to be 0 - probably error in backend calculation.
        rowExtrema.min = 0;
        let rowBinWidth     = (rowExtrema.max - rowExtrema.min) / this._binCounts.y;
        let colBinWidth     = (colExtrema.max - colExtrema.min) / this._binCounts.x;

        // 1. Create dimension for sample-in-model measure vs. sample's DR model measure.
        this._cf_dimensions[attribute] = this._crossfilter.dimension(
            // Determine bin number and return as value (i. e. we represent the bin number instead
            // of the real, rounded value).
            function(d) {
                // (a) Get row number.
                let rowValue = d[rowAttribute];
                if (rowValue <= rowExtrema.min)
                    rowValue = rowExtrema.min;
                else if (rowValue >= rowExtrema.max) {
                    rowValue = rowExtrema.max - rowBinWidth;
                }

                // (b) Get column number.
                let colValue = d[colAttribute];
                if (colValue <= colExtrema.min)
                    colValue = colExtrema.min;
                else if (colValue >= colExtrema.max) {
                    colValue = colExtrema.max - colBinWidth;
                }

                let binnedColValue  = colValue / colBinWidth;
                let binnedRowValue  = rowValue / rowBinWidth;

                // Calculate bin index and use as column/row number.
                // Use round() instead of floor, if record is a dummy (to ensure proper rounding for bin placmeent).
                return [
                    d.model_id !== -1 ? +Math.floor(binnedColValue) : +Math.round(binnedColValue),
                    d.model_id !== -1 ? +Math.floor(binnedRowValue) : +Math.round(binnedRowValue)
                ];
            }
        );

        // 2. Define group as sum of intersection sample_id/model_id - since only one element per
        // group exists, sum works just fine.
        this._cf_groups[attribute] = this._cf_dimensions[attribute].group().reduceCount();

        // 3. Find extrema.
        this._calculateHistogramExtremaForAttribute(attribute);
    }

    /**
     * Initializes singular dimensions w.r.t. histograms.
     */
    _initHistogramDimensionsAndGroups()
    {
        let yAttribute                      = null;
        let extrema                         = {min: 0, max: 1};
        let histogramAttribute              = null;
        let binWidth                        = null;

        // -----------------------------------------------------
        // 1. Create group for histogram on x-axis (SIMs).
        // -----------------------------------------------------

        yAttribute                          = "measure";
        extrema                             =  this._cf_extrema[yAttribute];
        histogramAttribute                  = "samplesInModels#" + yAttribute;
        this._binWidths[histogramAttribute] = (extrema.max - extrema.min) / this._binCounts.x;
        binWidth                            = this._binWidths[histogramAttribute];

        // Form group.
        this._cf_groups[histogramAttribute] = this._cf_dimensions[yAttribute]
            .group(function(value) {
                if (value <= extrema.min)
                    value = extrema.min;
                else if (value >= extrema.max) {
                    value = extrema.max - binWidth;
                }

                return Math.floor(value / binWidth) * binWidth;
            })
            // Custome reducer ignoring dummys with model_id === -1.
            .reduce(
                function(counter, item) {
                    return item.model_id !== -1 ? ++counter : counter;
                },
                function(counter, item) {
                    return item.model_id !== -1 ? --counter : counter;
                },
                function() {
                    let counter =0;
                    return counter;
                }
            );

        // CONTINUE HERE. Idea: Offer sorting options
        //  (1) Rows and orders separately - offer sorting by natural order (i. e. quality values) or sorting by
        //      frequency (i. e. sorting by numbers of SIMs in this group).
        //      Note that UI has to offer options separately by axes.
        //  (2) Cluster sorting - how exactly? Cluster cells of similar value/color? Discuss.
        //
        // TECHNICAL REALIZATION:
        //      Add one value per sorting option to group reflecting at which position this group should be.
        //      Use a lookup dict to return the "correct" key (which is the actual value from the natural sort order,
        //      i. e. between 0 and 1, so we don't have to change the chart's range etc.).
        //      Values are calculated by DissonanceDataset. Note that values have to be recalculated in case cross-
        //      operator brushing + linking is to be implemented.
        //      Then, in processSettingsChanges(), key accessor is set to corresponding value - directly or indirectly
        //      by just updating the instance attribute used for the lookup dict. Charts have to be re-rendered.
        //      Note that label function has to be updated as well to reflect the true value as opposed to the currently
        //      active one that's just used for placement purposes.
        //
        //      Consider also that changes on one axis can influence sort order on other axis, so a recalculation after
        //      any brushing action is necessary anyway (even w/o cross-op. b+l). -> Sort groups by values and update
        //      lookup dict.
        //
        //      CAVEAT: Does association of items with bins work correctly with this approach? What are possible
        //      alternative approaches, since chart.ordering() does not seem to work?
        //      -> Post on SO! Might save a lot of trouble.
        //
        //      Safer approach: Instead of showing range from extrema.min to extrema.max, use a range from 1 to
        //      binCount - keep bin value information in group and show in labels, if desired.
        //      Analogous approach to heatmap binning and should ensure correct B+L process in dc.js/crossfilter.js
        //      framework, since no key faking is going on (just group key re-assignments).
        //      Hence, each group would carry it's key (sort index), a value (number of records) and it's property
        //      threshold (min. value to be associated with this group).
        //
        //      Probably the cleanest way: Redefine .group() function (with external data structs. like dicts to keep
        //      track of _all_ the information) - use original group as base. When data is grouped in original group,
        //      keep track of which group key -> which sort index (can be computed after grouping with sort of copy).
        //      Intermediate: Which group key -> which data item. Then sort and compute aggregate information.
        //      Then, in dedicated sort group (one per sort order): Use external information computed in first group
        //      (i. e. which "normal" group has which sort index) to group data.
        //      Then update chart and set group with corresponding ordering as group for chart.
        //      Caveat: Grouping has to be performed in desired order (perhaps manually).

        for (let group of this._cf_groups[histogramAttribute].all()) {
            group["blub"] ="bla";
        }
        console.log(this._cf_groups[histogramAttribute].all());

        // Calculate extrema.
        this._calculateHistogramExtremaForAttribute(histogramAttribute);

        // -----------------------------------------------------
        // 2. Create group for histogram on y-axis (number of
        //    samples-in-models with given DR modelmeasure).
        // -----------------------------------------------------

        yAttribute                          = this._supportedDRModelMeasure;
        extrema                             =  this._cf_extrema[yAttribute];
        histogramAttribute                  = "samplesInModels#" + yAttribute;
        this._binWidths[histogramAttribute] = (extrema.max - extrema.min) / this._binCounts.y;
        binWidth                            = this._binWidths[histogramAttribute];

        // Form group.
        this._cf_groups[histogramAttribute] = this._cf_dimensions[yAttribute]
            .group(function(value) {
                if (value <= extrema.min)
                    value = extrema.min;
                else if (value >= extrema.max) {
                    value = extrema.max - binWidth;
                }

                return Math.floor(value / binWidth) * binWidth;
            })
            // Custome reducer ignoring dummys with model_id === -1.
            .reduce(
                function(counter, item) {
                    return item.model_id !== -1 ? ++counter : counter;
                },
                function(counter, item) {
                    return item.model_id !== -1 ? --counter : counter;
                },
                function() {
                    let counter = 0;
                    return counter;
                }
            );

        // Calculate extrema.
        this._calculateHistogramExtremaForAttribute(histogramAttribute);
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
            let countA = entryA.value;
            let countB = entryB.value;

            return countA > countB ? 1 : (countB > countA ? -1 : 0);
        });

        // Determine extrema.
        this._cf_extrema[histogramAttribute] = {
            min: (sortedData[0].value),
            max: (sortedData[sortedData.length - 1].value)
        };

        // Update extrema by padding values (hardcoded to 10%) for x-axis.
        this._cf_intervals[histogramAttribute] =
            this._cf_extrema[histogramAttribute].max -
            this._cf_extrema[histogramAttribute].min;
    }
}