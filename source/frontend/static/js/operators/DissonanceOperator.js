import Operator from "./Operator.js";
import DissonancePanel from "../panels/DissonancePanel.js";
import DissonanceSettingsPanel from "../panels/settings/DissonanceSettingsPanel.js";


/**
 * Class for dissonance operators (i. e. for showing disagreement between generated
 * models on particular instances/local scale).
 * One operator operates on exactly one dataset (-> one instance of class DRMetaDataset).
 */
export default class DissonanceOperator extends Operator
{
    /*
     * Constructs new DissonanceOperator.
     * @param name
     * @param stage
     * @param dataset Instance of DRMetaDataset class.
     * @param parentDivID
     */
    constructor(name, stage, dataset, parentDivID)
    {
        super(name, stage, "1", "0", dataset, parentDivID);

        // Update involved CSS classes.
        $("#" + this._target).addClass("dissonance-operator");

        // Construct all necessary panels.
        this.constructPanels();
    }

    /**
     * Constructs all panels required by this operator.
     */
    constructPanels()
    {
        // 1. Construct panel for surrogate model visualization.
        let dissPanel = new DissonancePanel(
            "Pointwise Quality",
            this
        );
        this._panels[dissPanel.name] = dissPanel;

        // 2. Construct panel for settings.
        let settingsPanel = new DissonanceSettingsPanel(
            "Sample Dissonance: Settings", this, null, "dissonance-info-settings-icon"
        );
        this._panels[settingsPanel.name] = settingsPanel;
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

    filter(embeddingIDs)
    {
        this._dataset._cf_dimensions["model_id"].filter(id => embeddingIDs.has(id));
        this._panels["Pointwise Quality"].render();
    }
}