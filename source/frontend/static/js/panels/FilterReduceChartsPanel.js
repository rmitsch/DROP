import Panel from "./Panel.js";
import Utils from "../Utils.js";
import ParetoScatterplot from "../charts/ParetoScatterplot.js";
import Histogram from "../charts/Histogram.js";

/**
 * Panel holding scatterplots and histograms in operator FilterReduce.
 */
export default class FilterReduceChartsPanel extends Panel
{
    /**
     * Constructs new FilterReduce charts panel.
     * @param name
     * @param operator
     * @param parentDivID
     */
    constructor(name, operator, parentDivID) {
        super(name, operator, parentDivID);

        // Create div structure for child nodes.
        let divStructure        = this.createDivStructure();
        this._containerDivIDs   = divStructure.containerDivIDs;
        this._histogramDivIDs   = divStructure.histogramDivIDs;

        // Generate charts.
        this.generateCharts();

        // Render all charts in prototype stage (work with groups instead of using .renderAll()!).
        console.log("Rendering...");
        dc.renderAll();
        console.log("Finished rendering.");
    }

    /**
     * Generates all chart objects. Does _not_ render them.
     */
    generateCharts()
    {
        console.log("Generating...");

        // Define style options for charts.
        let histogramStyle = {
            showAxisLabels: false,
            height: 60,
            width: 100,
            excludedColor: "#ccc",
            numberOfTicks: {
                x: 2,
                y: 0
            },
            showTickMarks: true
        };

        // Define style options for charts.
        let scatterplotStyle = {
            showAxisLabels: false,
            height: 100,
            width: 100,
            symbolSize: 1.5,
            excludedOpacity: 1,
            excludedSymbolSize: 1.5,
            excludedColor: "#ccc",
            numberOfTicks: {
                x: 2,
                y: 0
            },
            showTickMarks: true
        };

        // Fetch reference to dataset.
        let dataset = this._operator._dataset;

        // -----------------------------------
        // 0. Histograms.
        // -----------------------------------

        // Unfold names of hyperparamater objects in list.
        let hyperparameters = Utils.unfoldHyperparameterObjectList(dataset.metadata.hyperparameters);
        // Iterate over all attributes.
        for (let attribute of hyperparameters.concat(dataset.metadata.objectives)) {
            // Generate histogram.
            let histogram = new Histogram(
                attribute + ".histogram",
                this,
                [attribute],
                dataset,
                histogramStyle,
                // Place chart in previously generated container div.
                this._histogramDivIDs[attribute]
            );
        }

        // -----------------------------------
        // 1. Hyperparameter-objective combinations.
        // -----------------------------------

        // Iterate over hyperparameter.
        for (let hyperparameter of dataset.metadata.hyperparameters) {
            // Iterate over objectives.
            for (let objective of dataset.metadata.objectives) {
                // Don't display categorical values as scatterplot.
                if (hyperparameter.type !== "categorical") {
                    let scatterplot = new ParetoScatterplot(
                        hyperparameter.name + ":" + objective,
                        this,
                        [hyperparameter.name, objective],
                        dataset,
                        scatterplotStyle,
                        // Place chart in previously generated container div.
                        this._containerDivIDs[hyperparameter.name]
                    );
                }
            }
        }

        // -----------------------------------
        // 2. Objective-objective combinations.
        // -----------------------------------

        // Iterate over objectives.
        for (let i = 0; i < dataset.metadata.objectives.length; i++) {
            let objective1 = dataset.metadata.objectives[i];

            // Temporary: Fill slot with empty div.
            for (let k = 0; k <= i; k++) {
                let div         = document.createElement('div');
                div.className   = 'chart-placeholder';
                $("#" + this._containerDivIDs[objective1]).append(div);
            }

            // Iterate over objectives.
            for (let j = i + 1; j < dataset.metadata.objectives.length; j++) {
                let objective2 = dataset.metadata.objectives[j];

                let scatterplot = new ParetoScatterplot(
                    objective1 + ":" + objective2,
                    this,
                    [objective1, objective2],
                    dataset,
                    scatterplotStyle,
                    // Place chart in previously generated container div.
                    this._containerDivIDs[objective1]
                );
            }
        }

        console.log("Finished generating.");
    }

    /**
     * Create (hardcoded) div structure for child nodes.
     * @returns {Object}
     */
    createDivStructure()
    {
        let containerDivIDs = {};
        let histogramDivIDs = {};
        let dataset         = this._operator._dataset;

        // Create row labels.
        let labelContainer = Utils.spawnChildDiv(this._target, null, 'filter-reduce-labels-container');
        // Add labels to container.
        for (let objective of dataset.metadata.objectives) {
            let label = Utils.spawnChildDiv(labelContainer.id, null, 'filter-reduce-row-label');
            Utils.spawnChildSpan(label.id, null, 'filter-reduce-row-label-text', objective);
        }

        // -----------------------------------
        // Create container divs.
        // -----------------------------------

        // Unfold names of hyperparamater objects in list.
        let hyperparameters = Utils.unfoldHyperparameterObjectList(dataset.metadata.hyperparameters);

        // Iterate over all attributes.
        for (let attribute of hyperparameters.concat(dataset.metadata.objectives)) {
            let div = Utils.spawnChildDiv(this._target, null, "filter-reduce-charts-container");
            containerDivIDs[attribute] = div.id;

            // Add column label.
            Utils.spawnChildDiv(div.id, null, "title", attribute);

            // Add div for histogram.
            histogramDivIDs[attribute] = Utils.spawnChildDiv(div.id, null, "histogram").id;
        }

        return {containerDivIDs: containerDivIDs, histogramDivIDs: histogramDivIDs};
    }
}