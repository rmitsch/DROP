import SettingsPanel from "./SettingsPanel.js";
import Utils from "../../Utils.js";

/**
 * Class for dissonance settings panel.
 */
export default class DissonanceSettingsPanel extends SettingsPanel {
    /**
     * Constructs new settings panel for dissonance operator.
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

        // Add <select> for selection of sorting order.
        settingsHTML += "<div class='setting-option'>";
        settingsHTML += "<span id='dissonance-settings-sort-order'>Sorting order</span>";
        settingsHTML += "<select>" +
            "  <option value='sim-quality'>By sample-in-model quality</option>" +
            "  <option value='m-quality'>By model quality</option>" +
            "  <option value='cluster'>By clusters</option>" +
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