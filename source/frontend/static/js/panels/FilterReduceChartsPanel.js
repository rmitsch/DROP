import Panel from "./Panel.js";
import Utils from "../Utils.js";
import ParetoScatterplot from "../charts/ParetoScatterplot.js";

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
        this._containerDivIDs = this.createDivStructure();

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
        let style = {
            showAxisLabels: false,
            height: 100,
            width: 100,
            symbolSize: 1.5,
            excludedOpacity: 0.1,
            numberOfTicks: {
                x: 2,
                y: 0
            },
            showTickMarks: true
        };

        // Fetch reference to dataset.
        let dataset = this._operator._dataset;

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
                        style,
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
                    style,
                    // Place chart in previously generated container div.
                    this._containerDivIDs[objective1]
                );
            }
        }

        console.log("Finished generating.");
    }

    /**
     * Create (hardcoded) div structure for child nodes.
     * @returns {object}
     */
    createDivStructure()
    {
        let hyperparameters = [];
        let containerDivIDs = {};
        let dataset         = this._operator._dataset;

        // Unfold names of hyperparamater objects in list.
        for (let hyperparam in dataset.metadata.hyperparameters) {
            hyperparameters.push(dataset.metadata.hyperparameters[hyperparam].name);
        }

        // Create container divs.
        for (let attribute of hyperparameters.concat(dataset.metadata.objectives)) {
            let div         = document.createElement('div');
            div.id          = Utils.uuidv4();
            div.className   = 'filter-reduce-charts-container';
            $("#" + this._target).append(div);

            let labelDiv        = document.createElement('div');
            labelDiv.id         = Utils.uuidv4();
            labelDiv.className  = 'title';
            labelDiv.innerHTML  = attribute;
            $("#" + div.id).append(labelDiv);

            // Add div ID to dictionary.
            containerDivIDs[attribute] = div.id;
        }

        return containerDivIDs;
    }
}