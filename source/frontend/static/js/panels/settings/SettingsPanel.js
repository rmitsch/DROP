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

        // Update involved CSS classes.
        $("#" + this._target).addClass("settings-panel");

        // Create div structure for child nodes.
        this._divStructure = this._createDivStructure();

        // Set click listener.
        let scope = this;
        $("#" + iconID).click(function() {
            $("#" + scope._target).dialog({
                title: scope._name,
                width: $("#" + scope._operator._stage._target).width() / 4,
                height: $("#" + scope._operator._stage._target).height() / 2
            });
        });

        // Make class abstract.
        if (new.target === SettingsPanel) {
            throw new TypeError("Cannot construct SettingsPanel instances.");
        }
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