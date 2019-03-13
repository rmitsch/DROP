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
        this._name              = name;
        this._operator          = operator;
        this._charts            = {};
        this._target            = Utils.uuidv4();
        // Store size of panel at time of last render.
        this._lastPanelSize     = {width: 0, height: 0};

        // Panels datasets never differ from their operators'.
        this._data      = this._operator.data;
        this._metadata  = this._operator.metadata;

        // Create div structure for this panel.
        let div         = document.createElement('div');
        div.id          = this._target;
        div.className   = 'panel';
        $("#" + ((typeof parentDivID == "undefined" || parentDivID === null) ? this._operator.target : parentDivID)).append(div);

        // Make class abstract.
        if (new.target === Panel) {
            throw new TypeError("Cannot construct Panel instances.");
        }
    }

    /**
     * Generates all chart objects. Does _not_ render them.
     */
    _generateCharts()
    {
        throw new TypeError("Panel._generateCharts(): Abstract method must not be called.");
    }

    /**
     * (Re-)Renders all charts in this panel.
     */
    render()
    {
        throw new TypeError("Panel.render(): Abstract method must not be called.");
    }

    /**
     * Resizes all charts in this panel.
     */
    resize()
    {
        throw new TypeError("Panel.resize(): Abstract method must not be called.");
    }

    /**
     * Updates current filtering by specifying which IDs are to be considered active.
     * Triggered by filter() operation for operator.
     * Note:
     * @param embeddingIDs All active embedding IDs.
     */
    filter(embeddingIDs)
    {
        throw new TypeError("Panel.filter(): Abstract method must not be called.");
    }

    /**
     * Processes changes in data or configuration caused by changes in settings.
     * @param delta New/updated dataset or configuration.
     */
    processSettingsChange(delta)
    {
        throw new TypeError("Panel.processSettingsChange(): Abstract method must not be called.");
    }

    /**
     * Highlights data point in all charts in this panel.
     * @param id
     * @param source
     * @param propagate Determines whether highlight event should be propagated to stage.
     */
    highlight(id, source, propagate = false)
    {
        for (let key in this._charts) {
            if (this._charts[key]._name !== source)
                this._charts[key].highlight(id, source);
        }

        if (propagate)
            this._operator.highlight(id, this._name, propagate);
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