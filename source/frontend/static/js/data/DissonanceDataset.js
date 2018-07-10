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
     */
    constructor(name, data, binCounts)
    {
        super(name, data);

        this._axisPaddingRatio  = 0;
        this._binCounts         = binCounts;
        this._binWidths         = {};

        // Count number of samples and models.
        this._recordCounts = this._countRecordIDs();

        // todo
        // * Sample ID to sample name mapping
        // * UI: Search field
        // * Create dimensions and groups.
        // * Integrate DissonanceDataset in DissonanceChart.

        // Set up containers for crossfilter data.
        this._crossfilter = crossfilter(this._data);

        // Initialize crossfilter data.
        this._initSingularDimensionsAndGroups();
        this._initHistogramDimensionsAndGroups();
        this._initBinaryDimensionsAndGroups();
    }

    /**
     * Count how many models/samples are in the current dataset.
     * @returns {{model_id: number, sample_id: number}}
     * @private
     */
    _countRecordIDs()
    {
        let modelIDs = new Set();
        let sampleIDs = new Set();
        for (let record of this._data) {
            if (!(record.model_id in modelIDs))
                modelIDs.add(record.model_id);
            if (!(record.sample_id in sampleIDs))
                sampleIDs.add(record.sample_id);
        }

        return {
            model_id: modelIDs.size,
            sample_id: sampleIDs.size
        };
    }

    _initSingularDimensionsAndGroups()
    {
        let attributes = ["sample_id", "model_id", "measure"];

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
        let attribute = "sample_id:model_id";

        // 1. Create dimension for samples vs. model.
        this._cf_dimensions[attribute] = this._crossfilter.dimension(
            function(d) {
                return [+d["sample_id"], +d["model_id"]];
            }
        );

        // 2. Define group as sum of intersection sample_id/model_id - since only one element per
        // group exists, sum works just fine.
        this._cf_groups[attribute + "#measure"] = this._cf_dimensions[attribute].group().reduceSum(
            function(d) {
                return d.measure;
            }
        );
    }

    /**
     * Initializes singular dimensions w.r.t. histograms.
     */
    _initHistogramDimensionsAndGroups()
    {
        let scope                           = this;
        let yAttribute                      = "measure";
        let extrema                         = {min: 0, max: 1};
        let histogramAttribute              = "sim#measure";
        this._binWidths[histogramAttribute] = (extrema.max - extrema.min) / this._binCounts.x;

        // todo
        // Binning:
        //  x. Add bin size argument in constructor.
        //  2. Bin values (store either in measureAvgs or binnedMeasureAvgs).
        //     Note that binning should happen in step 1 already (calc. avg.
        //     measure values) - and that avg. shouldn't be calculated before,
        //     but in custom reducer.
        //  3. Write custom reducer (calculating avg.) or just use reduction.avg() (https://github.com/crossfilter/reductio).
        //

        // -----------------------------------------------------
        // 1. Bin samples-in-models (SIMs) by quality measure
        // values.
        // -----------------------------------------------------
        //
        // for (let j = 0; j < this._data.length; j++) {
        //     let value   = this._data[j][yAttribute];
        //     if (value <= extrema.min)
        //         value = extrema.min;
        //     else if (value >= extrema.max) {
        //         value = extrema.max - binWidth * 1.1;
        //     }
        //
        //     this._data[j][binnedYAttribute] = Math.floor(value / binWidth) * binWidth;
        // }
        // // Add new dimension for binned values.
        // this._cf_dimensions[binnedYAttribute] = this._crossfilter.dimension(
        //         function(d) { return d[binnedYAttribute]; }
        // );

        // Create dimension for binned attribute.

        // -----------------------------------------------------
        // 1. Calculate sum of measure values per bin.
        // -----------------------------------------------------
        //
        // for (let record of this._data) {
        //     for (let attribute of xAttributes) {
        //         let value = record[attribute];
        //         if (!(value in measureAvgs[attribute]))
        //             measureAvgs[attribute][value] = record[yAttribute];
        //         else
        //             measureAvgs[attribute][value] += record[yAttribute];
        //     }
        // }
        //
        // // -----------------------------------------------------
        // // 2. Average measure sum over number of records.
        // // -----------------------------------------------------
        //
        // for (let attribute in measureAvgs) {
        //     for (let value in measureAvgs[attribute]) {
        //         measureAvgs[attribute][value] /= this._recordCounts[attribute];
        //     }
        // }

        // -----------------------------------------------------
        // 3. Create group for histogram on x-axis (SIMs).
        // -----------------------------------------------------
        // for (let xAttribute of xAttributes) {
        let binWidth = this._binWidths[histogramAttribute];

        // Form group.
        this._cf_groups[histogramAttribute] = this._cf_dimensions[yAttribute]
            .group(function(value) {
                if (value <= extrema.min)
                    value = extrema.min;
                else if (value >= extrema.max) {
                    value = extrema.max - binWidth * 0.9;
                }

                return Math.floor(value / binWidth) * binWidth;
            });

        // todo *** HISTOGRAM BUG metadata: Because data is ordered alphabetically instead of numerically? -> should happen with values >= 10.
        // https://stackoverflow.com/questions/25204782/sorting-ordering-the-bars-in-a-bar-chart-by-the-bar-values-with-dc-js
        this._cf_groups[histogramAttribute].order(function(d) { return +d.key; });

        // Calculate extrema.
        this._calculateHistogramExtremaForAttribute(histogramAttribute);
    }

    /**
     * Calculates extrema for all histogram dimensions/groups.
     * @param histogramAttribute
     */
    _calculateHistogramExtremaForAttribute(histogramAttribute)
    {
        // Calculate extrema for histograms.
        let sortedData          = this._cf_groups[histogramAttribute].all();

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
        this._cf_intervals[histogramAttribute] = this._cf_extrema[histogramAttribute].max - this._cf_extrema[histogramAttribute].min;
        // this._cf_extrema[histogramAttribute].min -= this._cf_intervals[histogramAttribute] / this._axisPaddingRatio;
        // this._cf_extrema[histogramAttribute].max += this._cf_intervals[histogramAttribute] / this._axisPaddingRatio;
    }
}