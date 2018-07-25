import SettingsPanel from "./SettingsPanel.js";
import Utils from "../../Utils.js";

/**
 * Class for surrogate model settings panel.
 */
export default class SurrogateModelSettingsPanel extends SettingsPanel {
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
        let settingsHTML = "";

        // -----------------------------------
        // 1. Generate HTML for setting
        //    options.
        // -----------------------------------

        // Add range control for tree depth.
        settingsHTML += "<div class='setting-option'>";
        settingsHTML += "<span id='surrogate-settings-tree-depth-label'>Tree Depth</span>";
        settingsHTML += "<div class='range-control'>" +
            "<datalist id='tickmarks'>" +
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
            "<input type='range' list='tickmarks'>" +
            "</div>";
        settingsHTML += "</div>";

        // Add <select multiple> for selection of target objective(s).
        settingsHTML += "<div class='setting-option'>";
        settingsHTML += "<span id='surrogate-settings-target-objective'>Target objective</span>";
        settingsHTML += "<select multiple>" +
            "  <option value='runtime'>Runtime</option>" +
            "  <option value='rnx'>R<sub>nx</sub></option>" +
            "  <option value='bnx'>B<sub>nx</sub></option>" +
            "  <option value='stress'>Stress</option>" +
            "  <option value='accuracy'>Accuracy</option>" +
        "</select>";
        settingsHTML += "</div>";

        // -----------------------------------
        // 2. Create title and options container.
        // -----------------------------------

        // Note: Listener for table icon is added by FilterReduceOperator, since it requires information about the table
        // panel.
        $("#" + this._target).html(
            "<div class='settings-content'>" + settingsHTML + "</div>" +
            "<button class='pure-button pure-button-primary settings-update-button'>Apply changes</button>"
        );

        return {
            content: this._target
        };
    }
}