import SettingsPanel from "./SettingsPanel.js";
import Utils from "../../Utils.js";

/**
 * Class for filter reduce operator settings panel.
 */
export default class FilterReduceSettingsPanel extends SettingsPanel {
    /**
     * Constructs new settings panel for filter reduce operator.
     * @param name
     * @param operator
     * @param parentDivID
     * @param iconID
     */
    constructor(name, operator, parentDivID, iconID)
    {
        super(name, operator, parentDivID, iconID);
    }

    _createDivStructure() {
        // -----------------------------------
        // 1. Generate HTML for setting
        //    options.
        // -----------------------------------

        let settingsHTML = "";

        // Add range control for tree depth.
        settingsHTML += "<div class='setting-option'>";
        settingsHTML += "<span id='filter-reduce-settings-line-width'>Line width</span>";
        settingsHTML += "<div class='range-control'>" +
            "<datalist id='filter-reduce-tickmarks'>" +
            "  <option value='0.1' label='0.1'>" +
            "  <option value='0.2'>" +
            "  <option value='0.3'>" +
            "  <option value='0.4'>" +
            "  <option value='0.5' label='0.5'>" +
            "</datalist>" +
            "<input type='range' min='0.1' max='0.5' step='0.1' list='filter-reduce-tickmarks'>" +
            "</div>";

        settingsHTML += "</div>";

        // -----------------------------------
        // 2. Create title and options container.
        // -----------------------------------

        // Note: Listener for table icon is added by FilterReduceOperator, since it requires information about the table
        // panel.
        $("#" + this._target).html(
            "<div class='settings-content'>" + settingsHTML + "</div>" +
            "<button class='pure-button pure-button-primary settings-update-button' id='" + this._applyChangesButtonID + "'>Apply changes</button>"
        );

        return {
            content: this._target
        };
    }

    _applyOptionChanges()
    {
        console.log("applying changes")
    }
}