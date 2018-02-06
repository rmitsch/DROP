import Utils from "../Utils.js";

/**
 * A panel holds exactly one chart plus optional controls.
 * Panel is linked with exactly one operator.
 * Panels are separated through/contained in drag-panes.
 * Different panels can be linked, but this has to be done explicitly (as opposed to the automatic linking done by dc.js
 * for all charts inside a single panel).
 */
export default class Panel
{
    /**
     * Constructs new panel.
     * @param name
     * @param operator
     * @param parentDivID
     */
    constructor(name, operator, parentDivID)
    {
        this._name      = name;
        this._operator  = operator;
        this._charts    = {};
        this._target    = Utils.uuidv4();

        // Panels datasets never differ from their operators'.
        this._data      = this._operator.data;
        this._metadata  = this._operator.metadata;

        // Create div structure for this panel.
        let div         = document.createElement('div');
        div.id          = this._target;
        div.className   = 'panel';
        $("#" + (typeof parentDivID == "undefined" ? this._operator.target : parentDivID)).append(div);

        // Make class abstract.
        if (new.target === Panel) {
            throw new TypeError("Cannot construct Panel instances.");
        }
    }

    get name()
    {
        return this._name;
    }

    get charts()
    {
        return this._charts;
    }

    get operator()
    {
        return this._operator;
    }

    get target()
    {
        return this._target;
    }
}