import Operator from "./Operator.js";

/**
 * Class for ModelDetailOperator.
 * One operator operates on exactly one dataset (-> one instance of a DR model, including detailled information - like
 * coordinates - on all its records).
 */
export default class ModelDetailOperator extends Operator
{
    /**
     * Constructs new ModelDetailOperator.
     * Note that at initialization time no dataset is required.
     * @param name
     * @param stage
     * @param parentDivID
     */
    constructor(name, stage, parentDivID)
    {
        super(name, stage, "1", "n", null, parentDivID);

        // Update involved CSS classes.
        $("#" + this._target).addClass("model-detail-operator");

        // Construct all necessary panels.
        this.constructPanels();
    }

    /**
     * Constructs all panels required by this operator.
     */
    constructPanels()
    {
    //     // ----------------------------------------------
    //     // Generate panels.
    //     // ----------------------------------------------
    //
    //     // 1. Construct panels for charts.
    //     let frcPanel = new FilterReduceChartsPanel(
    //         "Hyperparameters & Objectives",
    //         this
    //     );
    //     this._panels[frcPanel.name] = frcPanel;
    //
    //     // 2. Construct panel for selection table.
    //     let tablePanel = new FilterReduceTablePanel(
    //         "Model Selection",
    //         this
    //     );
    //     this._panels[tablePanel.name] = tablePanel;
    //
    //     // 3. Construct panel for settings.
    //     let settingsPanel = new FilterReduceSettingsPanel(
    //         "Hyperparameters & Objectives: Settings", this, null, "filter-reduce-info-settings-icon"
    //     );
    //     this._panels[settingsPanel.name] = settingsPanel;
    //
    //     // ----------------------------------------------
    //     // Configure modals.
    //     // ----------------------------------------------
    //
    //     let scope = this;
    //
    //     // 4. Set click listener for FRC panel's table modal.
    //     $("#filter-reduce-info-table-icon").click(function() {
    //         $("#" + scope._panels[tablePanel.name]._target).dialog({
    //             title: "All models",
    //             width: $("#" + scope._stage._target).width() / 2,
    //             height: $("#" + scope._stage._target).height() / 2
    //         });
    //     });
    }
}