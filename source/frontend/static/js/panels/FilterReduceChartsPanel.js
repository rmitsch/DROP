import Panel from "./Panel.js";
import Utils from "../Utils.js";
import ParetoScatterplot from "../charts/ParetoScatterplot.js";
import NumericalHistogram from "../charts/NumericalHistogram.js";
import CategoricalHistogram from "../charts/CategoricalHistogram.js";
import CategoricalParetoScatterplot from "../charts/CategoricalParetoScatterplot.js";
import ParallelCoordinates from "../charts/ParallelCoordinates.js"

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
        // dc.renderAll();
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
            height: 80,
            width: 120,
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
            height: 120,
            width: 120,
            symbolSize: 1,
            excludedOpacity: 1,
            excludedSymbolSize: 1,
            excludedColor: "#ccc",
            numberOfTicks: {
                x: 0,
                y: 2
            },
            numberOfXTicksInLastRow: 2,
            numberOfYTicksInFirstRow: 0,
            showTickMarks: true
        };

        // Fetch reference to dataset.
        let dataset = this._operator._dataset;

        // -----------------------------------
        // 1. Histograms.
        // -----------------------------------

        this._createHistograms(dataset, histogramStyle);

        // -----------------------------------
        // 2. Create scatterplots.
        // -----------------------------------

        this._createScatterplots(dataset, scatterplotStyle);


        console.log("Finished generating.");
    }

    /**
     * Creates histograms for this panel.
     * @param dataset
     * @param style
     * @private
     */
    _createHistograms(dataset, style)
    {
       // Iterate over all attributes.
        // Unfold names of hyperparamater objects in list.
        let hyperparameters = Utils.unfoldHyperparameterObjectList(dataset.metadata.hyperparameters);
        let attributes = hyperparameters.concat(dataset.metadata.objectives);
        for (let i = 0; i < attributes.length; i++) {
            let attribute   = attributes[i];
            let histogram   = null;
            // If attributes is objective or numerical hyperparameter: Spawn NumericalHistogram.
            // This is hacky and you should be ashamed of yourself.
            if (i < hyperparameters.length &&
                dataset.metadata.hyperparameters[i].type === "numeric" ||
                i >= hyperparameters.length) {
                // Generate numerical histogram.
                histogram = new NumericalHistogram(
                    attribute + ".histogram",
                    this,
                    [attribute],
                    dataset,
                    style,
                    // Place chart in previously generated container div.
                    this._histogramDivIDs[attribute]
                );
            }

            // Otherwise: Spawn categorical histogram.
            else {
                // Generate categorical histogram.
                histogram = new CategoricalHistogram(
                    attribute + ".histogram",
                    this,
                    [attribute],
                    dataset,
                    style,
                    // Place chart in previously generated container div.
                    this._histogramDivIDs[attribute]
                );
            }

            histogram.render();
        }
    }

    /**
     * Create histograms for this panel.
     * @param dataset
     * @param style
     * @private
     */
    _createScatterplots(dataset, style)
    {
        // -----------------------------------
        // 1. Hyperparameter-objective combinations.
        // -----------------------------------

        this._createHyperparameterObjectiveScatterplots(dataset, style, true);

        // -----------------------------------
        // 2. Objective-objective combinations.
        // -----------------------------------

        this._createObjectiveObjectiveScatterplots(dataset, style, true);
    }

    /**
     * Creates hyperparameter-objective scatterplots.
     * @param dataset
     * @param style
     * @param render Flag determining whether plots should be rendered immediately.
     * @private
     */
    _createHyperparameterObjectiveScatterplots(dataset, style, render)
    {
        // Iterate over hyperparameter.
        let hyperparameterIndex = 0;
        for (let hyperparameter of dataset.metadata.hyperparameters) {
            let objectiveIndex = 0;
            // Iterate over objectives.
            for (let objective of dataset.metadata.objectives) {
                // Adapt style settings, based on whether this is the first scatterplot or not.
                let updatedStyle                = $.extend(true, {}, style);
                updatedStyle.numberOfTicks.y    = hyperparameterIndex === 0 ? updatedStyle.numberOfTicks.y : updatedStyle.numberOfYTicksInFirstRow;
                updatedStyle.numberOfTicks.x    = objectiveIndex === dataset.metadata.objectives.length - 1 ? updatedStyle.numberOfXTicksInLastRow : updatedStyle.numberOfTicks.x;

                // Instantiate new scatterplot.
                let scatterplot = new ParetoScatterplot(
                    hyperparameter.name + ":" + objective,
                    this,
                    // If hyperparameter is categorical: Use suffix "*" to enforce usage of numerical
                    // representation.
                    [
                        hyperparameter.name + (hyperparameter.type === "categorical" ? "*" : ""),
                        objective
                    ],
                    dataset,
                    updatedStyle,
                    // Place chart in previously generated container div.
                    this._containerDivIDs[hyperparameter.name]
                );

                if (render)
                    scatterplot.render();

                    // let parCorPlot = new ParallelCoordinates(
                    //     hyperparameter.name + ":" + objective,
                    //     this,
                    //     [hyperparameter.name, objective],
                    //     dataset,
                    //     updatedStyle,
                    //     // Place chart in previously generated container div.
                    //     this._containerDivIDs[hyperparameter.name]
                    // );
                    //
                    // if (render)
                    //     parCorPlot.render();


                objectiveIndex++;
            }

            hyperparameterIndex++;
        }
    }

    /**
     * Creates objective-objective scatterplots.
     * @param dataset
     * @param style
     * @param render Flag determining whether plots should be rendered immediately.
     * @private
     */
    _createObjectiveObjectiveScatterplots(dataset, style, render)
    {
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

                // Adapt style settings, based on whether this is the first scatterplot or not.
                let updatedStyle                = $.extend(true, {}, style);
                updatedStyle.numberOfTicks.y    = 0;
                updatedStyle.numberOfTicks.x    = j === dataset.metadata.objectives.length - 1 ? updatedStyle.numberOfXTicksInLastRow : updatedStyle.numberOfTicks.x;


                // Instantiate new scatterplot.
                let scatterplot = new ParetoScatterplot(
                    objective1 + ":" + objective2,
                    this,
                    [objective1, objective2],
                    dataset,
                    updatedStyle,
                    // Place chart in previously generated container div.
                    this._containerDivIDs[objective1]
                );
                if (render)
                    // Render scatterplot.
                    scatterplot.render();
            }
        }
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

        return {
            containerDivIDs: containerDivIDs,
            histogramDivIDs: histogramDivIDs
        };
    }
}