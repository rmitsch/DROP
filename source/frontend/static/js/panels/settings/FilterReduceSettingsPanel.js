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

    _createDivStructure()
    {
        // -----------------------------------
        // 1. Generate HTML for setting
        //    options.
        // -----------------------------------

        let settingsHTML = "";

        // Add range control for tree depth.
        settingsHTML += "<div class='setting-option'>";

        settingsHTML += "<div class='settings-section-header'>Line opacity</div>";
        // Bin fraction.
        settingsHTML += "<div class='settings-subsection'>";
        settingsHTML += "<span id='filter-reduce-settings-line-opacity-bin-fraction'>Number of bins</span>";
        settingsHTML += "<input type='number' id='filter-reduce-bin-number' min='1' max='10000' step='1' value='1'>";
        settingsHTML += "</div>";

        // Logarithmic scaling.
        settingsHTML += "<div class='settings-subsection'>";
        settingsHTML += "<span id='filter-reduce-settings-line-opacity-log-scaling'>Use log. scaling</span>";
        settingsHTML += "<input id='filter-reduce-settings-line-log-scale' type='checkbox' checked>"
        settingsHTML += "</div>";

        settingsHTML += "</div>";

        // -----------------------------------
        // 2. Create title and options container.
        // -----------------------------------

        // Note: Listener for table icon is added by parent class, since it requires information about the table
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
        this._operator.propagateSettingsChanges(
            {
                binFraction: parseInt($("#filter-reduce-bin-number").val()),
                useLogs: $("#filter-reduce-settings-line-log-scale").is(":checked")
            },
            this._name
        )
    }
}