import SettingsPanel from "./SettingsPanel.js";
import Utils from "../../Utils.js";

/**
 * Class for surrogate model settings panel.
 */
export default class SurrogateModelSettingsPanel extends SettingsPanel
{
    /**
     * Constructs new settings panel for surrogate model operator.
     * @param name
     * @param operator
     * @param parentDivID
     * @param iconID
     */
    constructor(name, operator, parentDivID, iconID)
    {
        super(name, operator, parentDivID, iconID);
    }

    _createDivStructure()
    {
        let settingsHTML    = "";
        let scope           = this;

        // -----------------------------------
        // 1. Generate HTML for setting
        //    options.
        // -----------------------------------

        // Add range control for tree depth.
        settingsHTML += "<div class='setting-option'>";
        settingsHTML += "<span id='surrogate-settings-tree-depth-label'>Tree Depth</span>";
        settingsHTML += "<div class='range-control'>" +
            "<datalist id='surrogate-tickmarks'>" +
            "  <option value='1' label='0'>" +
            "  <option value='2'>" +
            "  <option value='3'>" +
            "  <option value='4'>" +
            "  <option value='5' label='5'>" +
            "  <option value='6'>" +
            "  <option value='7'>" +
            "  <option value='8'>" +
            "  <option value='9'>" +
            "  <option value='10' label='10'>" +
            "</datalist>" +
            "<input type='range' min='1' max='10' step='1' id='surrogate-settings-tree-depth-range' list='surrogate-tickmarks'>" +
            "</div>";
        settingsHTML += "</div>";

        // Add <select multiple> for selection of target objective(s).
        settingsHTML += "<div class='setting-option'>";
        settingsHTML += "<span id='surrogate-settings-target-objective'>Target objective</span>";
        settingsHTML += "<select multiple id='surrogate-settings-target-objective-select'>" +
            "  <option value='runtime'>Runtime</option>" +
            "  <option value='r_nx'>R<sub>nx</sub></option>" +
            "  <option value='stress'>Stress</option>" +
            "  <option value='classification_accuracy'>Classification accuracy</option>" +
            "  <option value='separability_metric'>Separability metric</option>" +
        "</select>";
        settingsHTML += "</div>";

        // -----------------------------------
        // 2. Create title and options container.
        // -----------------------------------

        $("#" + this._target).html(
            "<div class='settings-content'>" + settingsHTML + "</div>" +
            "<button class='pure-button pure-button-primary settings-update-button' id='" + scope._applyChangesButtonID + "'>Apply changes</button>"
        );

        return {
            content: this._target
        };
    }

    /**
     * Extracts currently chosen settings for surrogate model.
     * @returns {{treeDepth, objectives: *}}
     */
    static getOptionValues()
    {
        let treeDepth   = $("#surrogate-settings-tree-depth-range")[0].value;
        let objectives  = $("#surrogate-settings-target-objective-select").val();

        // If no objective has been chosen: Use all.
        if (objectives.length === 0)
            objectives = ["runtime", "r_nx", "stress", "classification_accuracy", "separability_metric"];

        // Concatenate objectives.
        let objectiveString = objectives[0];
        if (objectives.length > 1) {
            for (let i = 1; i < objectives.length; i++)
                objectiveString += "," + objectives[i];
        }

        return {treeDepth: treeDepth, objectives: objectiveString}
    }

    _applyOptionChanges()
    {
        // -------------------------------------------------
        // 1. Extract option values.
        // -------------------------------------------------

        let options         = SurrogateModelSettingsPanel.getOptionValues();
        let objectiveString = options.objectives;
        let treeDepth       = options.treeDepth;

        // Assemble all filtered IDs.
        let idString = "";
        for (let id of this._operator.filteredIDs)
            idString += id + ",";
        idString = idString.substring(0, idString.length - 1);

        // -------------------------------------------------
        // 2. Fetch new surrogate model data.
        // -------------------------------------------------

        fetch("/get_surrogate_model_data?modeltype=tree&objs=" + objectiveString + "&depth=" + treeDepth + "&ids=" + idString,
            {
                headers: { "Content-Type": "application/json; charset=utf-8"},
                method: "GET"
            })
            .then(res => res.json())
            // -------------------------------------------------
            // 3. Update surrogate model chart with new data.
            // -------------------------------------------------
            .then(results => this._operator.propagateSettingsChanges(results, this._name));
    }

    processSettingsChange(delta)
    {
        // Do nothing (alt.: Show that settings have been propagated/updated).
    }
}