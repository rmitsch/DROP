import SettingsPanel from "./SettingsPanel.js";
import Utils from "../../Utils.js";

/**
 * Class for settings panels.
 * Accepts a map of options.
 */
export default class SurrogateModelSettingsPanel extends SettingsPanel {
    /**
     * Constructs new panel for charts for DissonanceOperator.
     * @param name
     * @param operator
     * @param parentDivID
     */
    constructor(name, operator, parentDivID)
    {
        super(name, operator, parentDivID);
    }

    _createDivStructure() {
        let scope = this;

        // -----------------------------------
        // 1. Generate HTML for setting
        //    options.
        // -----------------------------------

        let settingsHTML = "";

        settingsHTML += "<div class='setting-option'>" +
            "<span>Depth:</span>";

        settingsHTML += "<div class='range-control'>" +
            "<datalist id=\"tickmarks\">" +
            "  <option value=\"0\" label=\"0%\">" +
            "  <option value=\"10\">" +
            "  <option value=\"20\">" +
            "  <option value=\"30\">" +
            "  <option value=\"40\">" +
            "  <option value=\"50\" label=\"50%\">" +
            "  <option value=\"60\">" +
            "  <option value=\"70\">" +
            "  <option value=\"80\">" +
            "  <option value=\"90\">" +
            "  <option value=\"100\" label=\"100%\">" +
            "</datalist>" + "<input type=\"range\" list=\"tickmarks\">" +
            "</div>";
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