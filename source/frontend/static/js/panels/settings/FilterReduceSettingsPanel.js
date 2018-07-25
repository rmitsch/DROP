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