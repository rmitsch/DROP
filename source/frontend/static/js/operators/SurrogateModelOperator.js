import Operator from "./Operator.js";
import SurrogateModelPanel from "../panels/SurrogateModelPanel.js";
import SurrogateModelChart from "../charts/SurrogateModelChart.js";
import SettingsPanel from "../panels/settings/SettingsPanel.js";


/**
 * Class for SurrogateModel operators.
 * One operator operates on exactly one dataset (-> one instance of class DRMetaDataset).
 * See https://bl.ocks.org/ajschumacher/65eda1df2b0dd2cf616f.
 */
export default class SurrogateModelOperator extends Operator
{
    /**
     * Constructs new SurrogateModelOperator.
     * Note that SurrogateModelOperators is invariant towards which kernel was used for dimensionality reduction, since
     * all hyperparameter/objectives are defined in metadata. Thus, the specific operator name ("SurrogateModel:Tree")
     * is stored in a class attribute only (as opposed to branching the Operator class tree further).
     * @param name
     * @param stage
     * @param dataset Instance of DRMetaDataset class.
     * @param modelType Type of model to be used as surrogate. Currently available: Decision tree.
     * @param parentDivID
     */
    constructor(name, stage, dataset, modelType, parentDivID)
    {
        // Relationship cardinality is 1:0, since one dataset is read and none is produced.
        super(name, stage, "1", "0", dataset, parentDivID);

        // Update involved CSS classes.
        $("#" + this._target).addClass("surrogate-model-operator");

        // Save which model (influences inference model and visualization)
        // should be used as surrogate - e. g. decision tree.
        this._modelType = modelType;

        // Construct all necessary panels.
        this.constructPanels();
    }

    /**
     * Constructs all panels required by this operator.
     */
    constructPanels()
    {
        // ----------------------------------------------
        // Generate panels.
        // ----------------------------------------------

        // 1. Construct panel for surrogate model visualization.
        let surrModelPanel = new SurrogateModelPanel(
            "Surrogate Model",
            this
        );
        this._panels[surrModelPanel.name] = surrModelPanel;

        // 2. Construct panel for settings.
        let settingsPanel = new SettingsPanel(
            "Surrogate Model: Settings",
            this,
            null,
            {
                "Target Objective": {
                    type: "dropdown",
                    values: ["Runtime", "R<sub>nx</sub>", "B<sub>nx</sub>", "Stress", "Accuracy", "Silhouette"],
                    default: "Runtime"
                },
                "Depth": {
                    type: "range",
                    range: [1, 10],
                    default: 5
                }
            }
        );
        this._panels[settingsPanel.name] = settingsPanel;

        // ----------------------------------------------
        // Configure modals.
        // ----------------------------------------------

        let scope = this;

        // 3. Set click listener for FRC panel's settings modal.
        $("#surrogate-info-settings-icon").click(function() {
            $("#" + scope._panels[settingsPanel.name]._target).dialog({
                title: "Settings",
                width: $("#" + scope._stage._target).width() / 4,
                height: $("#" + scope._stage._target).height() / 2
            });
        });
    }

    render()
    {
        for (let panelName in this._panels) {
            this._panels[panelName].render();
        }
    }

    resize()
    {
        for (let panelName in this._panels) {
            this._panels[panelName].resize();
        }
    }
}