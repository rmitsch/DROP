import uuidv4 from "../utils.js";

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
     * @param linked_crossfilter Reference to crossfilter instance. Might be null. If not null, specified crossfilter
     * instance is used (useful when multiple panels inside the same operator are supposed to operate on the same
     * dataset).
     */
    constructor(name, operator, linked_crossfilter)
    {
        this._name      = name;
        this._operator  = operator;
        this._charts    = {};
        this._target    = uuidv4();
        // Panels datasets never differ from their operators'.
        this._data      = this._operator.data;
        this._metadata  = this._operator.metadata;

        // Create div structure for this panel.
        let div         = document.createElement('div');
        div.id          = this._target;;
        div.className   = 'panel';
        $("#" + this._operator.target).append(div);

        // Instantiate crossfilter, if not done already.
        this._crossfilter = linked_crossfilter != null ? linked_crossfilter : crossfilter(this._data);

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