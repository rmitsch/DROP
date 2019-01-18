import Panel from "../Panel.js";
import Utils from "../../Utils.js";

/**
 * Abstract base class for settings panels.
 */
export default class SettingsPanel extends Panel
{
    /**
     * Constructor for abstract setting panel base class.
     * @param name
     * @param operator
     * @param parentDivID
     * @param iconID ID of icon used to launch options panel.
     */
    constructor(name, operator, parentDivID, iconID)
    {
        super(name, operator, parentDivID);

        // Define ID of "Apply changes" button.
        this._applyChangesButtonID = name
            .toLowerCase()
            .replace("&", "-")
            .replace(" ", "-")
            .replace(":", "-")
            .replace("--", "-")
            .replace("- ", "-").replace("- ", "-")
            .replace(" ", "-") +
            "-apply-changes-btn";

        // Update involved CSS classes.
        $("#" + this._target).addClass("settings-panel");

        // Create div structure for child nodes.
        this._divStructure = this._createDivStructure();

        // Set click listener.
        this._setClickListener(iconID);

        // Make class abstract.
        if (new.target === SettingsPanel) {
            throw new TypeError("Cannot construct SettingsPanel instances.");
        }
    }

    /**
     * Sets click listener for activation and updating charts.
     * @param iconID
     * @private
     */
    _setClickListener(iconID)
    {
        let scope   = this;
        const stage = $("#" + this._operator._stage._target);

        // Set listener for opening settings panel.
        $("#" + iconID).click(function() {
            $("#" + scope._target).dialog({
                title: scope._name,
                width: stage.width() / 4,
                height: stage.height() / 2
            });
        });

        // Set listener for parsing options and applying changes.
        $("#" + this._applyChangesButtonID).click(function() {
            scope._applyOptionChanges();
            $("#" + scope._target).dialog("close");
        });
    }

    /**
     * Apply new set of options (if different from previous set of options).
     * Instruct visualizations to update based on new data/configuration.
     * @private
     */
    _applyOptionChanges()
    {
        throw new TypeError("SettingsPanel._applyOptionChanges(): Abstract method must not be called.");
    }

    _generateCharts()
    {
        throw new TypeError("SettingsPanel._generateCharts(): Abstract method must not be called.");
    }

    /**
     * Create (hardcoded) div structure for child nodes.
     * @returns {Object}
     */
    _createDivStructure()
    {
        throw new TypeError("SettingsPanel._createDivStructure(): Abstract method must not be called.");
    }

    render()
    {
    }

    resize()
    {
    }
}