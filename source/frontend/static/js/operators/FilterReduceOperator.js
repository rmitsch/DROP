import Operator from "./Operator.js";
import FilterReduceChartsPanel from "../panels/FilterReduceChartsPanel.js";
import FilterReduceTablePanel from "../panels/FilterReduceTablePanel.js";
import FilterReduceSettingsPanel from "../panels/settings/FilterReduceSettingsPanel.js";

/**
 * Class for FilterReduceOperators with scattered (scree) and violin plots.
 * One operator operates on exactly one dataset (-> one instance of class DRMetaDataset).
 */
export default class FilterReduceOperator extends Operator
{
    /**
     * Constructs new FilterReduceOperator.
     * Note that FilterReduceOperator is invariant towards which kernel was used for dimensionality reduction, since all
     * hyperparameter/objectives are defined in metadata. Thus, the specific operator name ("FilterReduce:TSNE") is
     * stored in a class attribute only (as opposed to branching the Operator class tree further).
     * @param name
     * @param stage
     * @param dataset Instance of DRMetaDataset class.
     * @param parentDivID
     * @param tableParentDivID ID of parent div for embedding overview table.
     */
    constructor(name, stage, dataset, parentDivID, tableParentDivID)
    {
        super(name, stage, "1", "n", dataset, parentDivID);

        // Update involved CSS classes.
        $("#" + this._target).addClass("filter-reduce-operator");

        // Store ID of parent table div.
        this._tableParentDivID = tableParentDivID;

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

        // 1. Construct panel for selection table.
        let tablePanel = new FilterReduceTablePanel(
            "Model Selection",
            this,
            this._tableParentDivID
        );
        this._panels[tablePanel.name] = tablePanel;

        // 2. Construct panels for charts.
        let frcPanel = new FilterReduceChartsPanel(
            "Hyperparameters & Objectives",
            this
        );
        this._panels[frcPanel.name] = frcPanel;

        // 3. Construct panel for settings.
        let settingsPanel = new FilterReduceSettingsPanel(
            "Hyperparameters & Objectives: Settings",
            this,
            null,
            "filter-reduce-info-settings-icon"
        );
        this._panels[settingsPanel.name] = settingsPanel;
    }

    get tablePanel()
    {
        return this._panels["Model Selection"];
    }

    resize()
    {
        for (let panelName in this._panels) {
            this._panels[panelName].resize();
        }
    }

    render()
    {
        for (let panelName in this._panels) {
            this._panels[panelName].render();
        }
    }

    propagateSettingsChanges(delta, sourcePanelName)
    {
        // We know that currently only one panel is affected by setting changes.
        // Otherwise changes would have to be propagated as done in parent method.
        this._panels["Hyperparameters & Objectives"].updateSSPBinning(delta);
    }

    filter(embeddingIDs)
    {
        this._panels["Hyperparameters & Objectives"].updateFilteredRecordBuffer(embeddingIDs);

        // Assuming dimensions have already been filtered.
        // for (let panelName in this._panels) {
        //     this._panels[panelName].render();
        // }
    }
}