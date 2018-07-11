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
        this._complementDatasetWithDRMeasure();
        console.log(this._data);
//
//         this._data = [
// 	{model_id: 0, sample_id: 0, measure: 4, r_nx: 18},
// 	{model_id: 1, sample_id: 0, measure: 16, r_nx: 3},
// 	{model_id: 2, sample_id: 0, measure: 9, r_nx: 3},
// 	{model_id: 3, sample_id: 0, measure: 16, r_nx: 13},
// 	{model_id: 4, sample_id: 0, measure: 11, r_nx: 14},
// 	{model_id: 5, sample_id: 0, measure: 8, r_nx: 16},
// 	{model_id: 6, sample_id: 0, measure: 17, r_nx: 12},
// 	{model_id: 7, sample_id: 0, measure: 11, r_nx: 7},
// 	{model_id: 8, sample_id: 0, measure: 3, r_nx: 2},
// 	{model_id: 9, sample_id: 0, measure: 14, r_nx: 18},
// 	{model_id: 10, sample_id: 0, measure: 16, r_nx: 5},
// 	{model_id: 11, sample_id: 0, measure: 6, r_nx: 0},
// 	{model_id: 12, sample_id: 0, measure: 0, r_nx: 1},
// 	{model_id: 13, sample_id: 0, measure: 13, r_nx: 5},
// 	{model_id: 14, sample_id: 0, measure: 3, r_nx: 19},
// 	{model_id: 15, sample_id: 0, measure: 2, r_nx: 1},
// 	{model_id: 16, sample_id: 0, measure: 18, r_nx: 0},
// 	{model_id: 17, sample_id: 0, measure: 10, r_nx: 9},
// 	{model_id: 18, sample_id: 0, measure: 18, r_nx: 15},
// 	{model_id: 19, sample_id: 0, measure: 10, r_nx: 16},
// 	{model_id: 20, sample_id: 0, measure: 17, r_nx: 13},
// 	{model_id: 21, sample_id: 0, measure: 0, r_nx: 1},
// 	{model_id: 22, sample_id: 0, measure: 13, r_nx: 14},
// 	{model_id: 23, sample_id: 0, measure: 1, r_nx: 9},
// 	{model_id: 24, sample_id: 0, measure: 4, r_nx: 15},
// 	{model_id: 25, sample_id: 0, measure: 19, r_nx: 6},
// 	{model_id: 26, sample_id: 0, measure: 3, r_nx: 1},
// 	{model_id: 27, sample_id: 0, measure: 17, r_nx: 4},
// 	{model_id: 28, sample_id: 0, measure: 19, r_nx: 18},
// 	{model_id: 29, sample_id: 0, measure: 15, r_nx: 16},
// 	{model_id: 30, sample_id: 0, measure: 15, r_nx: 3},
// 	{model_id: 31, sample_id: 0, measure: 19, r_nx: 7},
// 	{model_id: 32, sample_id: 0, measure: 17, r_nx: 10},
// 	{model_id: 33, sample_id: 0, measure: 14, r_nx: 10},
// 	{model_id: 34, sample_id: 0, measure: 14, r_nx: 7},
// 	{model_id: 35, sample_id: 0, measure: 2, r_nx: 4},
// 	{model_id: 36, sample_id: 0, measure: 1, r_nx: 12},
// 	{model_id: 37, sample_id: 0, measure: 18, r_nx: 5},
// 	{model_id: 38, sample_id: 0, measure: 15, r_nx: 6},
// 	{model_id: 39, sample_id: 0, measure: 8, r_nx: 2},
// 	{model_id: 40, sample_id: 0, measure: 2, r_nx: 19},
// 	{model_id: 41, sample_id: 0, measure: 9, r_nx: 18},
// 	{model_id: 42, sample_id: 0, measure: 11, r_nx: 17},
// 	{model_id: 43, sample_id: 0, measure: 16, r_nx: 4},
// 	{model_id: 44, sample_id: 0, measure: 7, r_nx: 7},
// 	{model_id: 45, sample_id: 0, measure: 10, r_nx: 17},
// 	{model_id: 46, sample_id: 0, measure: 16, r_nx: 19},
// 	{model_id: 47, sample_id: 0, measure: 7, r_nx: 18},
// 	{model_id: 48, sample_id: 0, measure: 1, r_nx: 15},
// 	{model_id: 49, sample_id: 0, measure: 1, r_nx: 15},
// 	{model_id: 50, sample_id: 0, measure: 19, r_nx: 18},
// 	{model_id: 51, sample_id: 0, measure: 6, r_nx: 8},
// 	{model_id: 52, sample_id: 0, measure: 14, r_nx: 1},
// 	{model_id: 53, sample_id: 0, measure: 5, r_nx: 8},
// 	{model_id: 54, sample_id: 0, measure: 19, r_nx: 19},
// 	{model_id: 55, sample_id: 0, measure: 15, r_nx: 3},
// 	{model_id: 56, sample_id: 0, measure: 8, r_nx: 14},
// 	{model_id: 57, sample_id: 0, measure: 0, r_nx: 9},
// 	{model_id: 58, sample_id: 0, measure: 15, r_nx: 16},
// 	{model_id: 59, sample_id: 0, measure: 15, r_nx: 10},
// 	{model_id: 60, sample_id: 0, measure: 4, r_nx: 1},
// 	{model_id: 61, sample_id: 0, measure: 16, r_nx: 7},
// 	{model_id: 62, sample_id: 0, measure: 18, r_nx: 14},
// 	{model_id: 63, sample_id: 0, measure: 10, r_nx: 16},
// 	{model_id: 64, sample_id: 0, measure: 6, r_nx: 15},
// 	{model_id: 65, sample_id: 0, measure: 8, r_nx: 13},
// 	{model_id: 66, sample_id: 0, measure: 4, r_nx: 15},
// 	{model_id: 67, sample_id: 0, measure: 16, r_nx: 19},
// 	{model_id: 68, sample_id: 0, measure: 3, r_nx: 4},
// 	{model_id: 69, sample_id: 0, measure: 5, r_nx: 9},
// 	{model_id: 70, sample_id: 0, measure: 10, r_nx: 19},
// 	{model_id: 71, sample_id: 0, measure: 18, r_nx: 10},
// 	{model_id: 72, sample_id: 0, measure: 8, r_nx: 11},
// 	{model_id: 73, sample_id: 0, measure: 14, r_nx: 8},
// 	{model_id: 74, sample_id: 0, measure: 14, r_nx: 17},
// 	{model_id: 75, sample_id: 0, measure: 11, r_nx: 2},
// 	{model_id: 76, sample_id: 0, measure: 17, r_nx: 16},
// 	{model_id: 77, sample_id: 0, measure: 3, r_nx: 10},
// 	{model_id: 78, sample_id: 0, measure: 9, r_nx: 8},
// 	{model_id: 79, sample_id: 0, measure: 18, r_nx: 13},
// 	{model_id: 80, sample_id: 0, measure: 9, r_nx: 11},
// 	{model_id: 81, sample_id: 0, measure: 13, r_nx: 9},
// 	{model_id: 82, sample_id: 0, measure: 1, r_nx: 13},
// 	{model_id: 83, sample_id: 0, measure: 19, r_nx: 3},
// 	{model_id: 84, sample_id: 0, measure: 3, r_nx: 12},
// 	{model_id: 85, sample_id: 0, measure: 12, r_nx: 16},
// 	{model_id: 86, sample_id: 0, measure: 18, r_nx: 0},
// 	{model_id: 87, sample_id: 0, measure: 8, r_nx: 13},
// 	{model_id: 88, sample_id: 0, measure: 7, r_nx: 4},
// 	{model_id: 89, sample_id: 0, measure: 11, r_nx: 10},
// 	{model_id: 90, sample_id: 0, measure: 6, r_nx: 18},
// 	{model_id: 91, sample_id: 0, measure: 4, r_nx: 2},
// 	{model_id: 92, sample_id: 0, measure: 19, r_nx: 3},
// 	{model_id: 93, sample_id: 0, measure: 6, r_nx: 8},
// 	{model_id: 94, sample_id: 0, measure: 12, r_nx: 18},
// 	{model_id: 95, sample_id: 0, measure: 1, r_nx: 3},
// 	{model_id: 96, sample_id: 0, measure: 16, r_nx: 7},
// 	{model_id: 97, sample_id: 0, measure: 19, r_nx: 15},
// 	{model_id: 98, sample_id: 0, measure: 17, r_nx: 3},
// 	{model_id: 99, sample_id: 0, measure: 5, r_nx: 17}
// ];
        // Set up containers for crossfilter data.
        this._crossfilter = crossfilter(this._data);

        // Initialize crossfilter data.
        this._initSingularDimensionsAndGroups();
        this._initHistogramDimensionsAndGroups();
        this._initBinaryDimensionsAndGroups();
    }

    /**
     * Adds model measures to internal dataset.
     * @private
     */
    _complementDatasetWithDRMeasure()
    {
        let drModelIDToMeasureValue = {};

        // Iterate over DR model indices, store corresponding measure.
        for (let drModelID in this._drModelMetadata._dataIndicesByID) {
            let drModelIndex                    = this._drModelMetadata._dataIndicesByID[drModelID];
            drModelIDToMeasureValue[drModelID]  = this._drModelMetadata._data[drModelIndex][this._supportedDRModelMeasure];
        }

        // Update internal dataset.
        for (let sampleInModelIndex in this._data) {
            this._data[sampleInModelIndex][this._supportedDRModelMeasure] = drModelIDToMeasureValue[
                this._data[sampleInModelIndex].model_id
            ];
            // this._data[sampleInModelIndex][this._supportedDRModelMeasure] = Math.floor(Math.random() * 20);
            // this._data[sampleInModelIndex].measure = Math.floor(Math.random() * 20);
        }

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
        let yAttribute                      = null;
        let extrema                         = {min: 0, max: 1};
        let histogramAttribute              = null;
        let binWidth                        = null;

        // -----------------------------------------------------
        // 1. Create group for histogram on x-axis (SIMs).
        // -----------------------------------------------------

        yAttribute                          = "measure";
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
            });

        // https://stackoverflow.com/questions/25204782/sorting-ordering-the-bars-in-a-bar-chart-by-the-bar-values-with-dc-js
        // Calculate extrema.
        this._calculateHistogramExtremaForAttribute(histogramAttribute);

        // -----------------------------------------------------
        // 2. Create group for histogram on y-axis (number of
        //    samples-in-models with given DR modelmeasure).
        // -----------------------------------------------------

        yAttribute                          = this._supportedDRModelMeasure;
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
            });

        // https://stackoverflow.com/questions/25204782/sorting-ordering-the-bars-in-a-bar-chart-by-the-bar-values-with-dc-js
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