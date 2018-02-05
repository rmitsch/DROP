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
     * @param targetDivID
     */
    constructor(name, operator, targetDivID) {
        super(name, operator, targetDivID);

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

        // Iterate over hyperparameter.
        for (let hyperparameter of dataset.metadata.hyperparameters) {

            // Iterate over objectives.
            for (let objective of dataset.metadata.objectives) {
                console.log(hyperparameter.name + ":" + objective);

                // Don't display categorical values as scatterplot.
                if (hyperparameter.type !== "categorical") {

                    let scatterplot = new ParetoScatterplot(
                        hyperparameter.name + ":" + objective,
                        this,
                        [hyperparameter.name, objective],
                        dataset,
                        style
                    );
                }
            }
        }

        dc.renderAll();

    }
}