import Utils from "../Utils.js";

/**
 * Abstract base class for operators.
 */
export default class Operator
{
    /**
     * Constructs new Operator.
     * @param name
     * @param stage
     * @param inputCardinality
     * @param outputCardinality
     * @param dataset Instance of DRMetaDataset class. Holds exactly one dataset.
     * @param parentDivID
     */
    constructor(name, stage, inputCardinality, outputCardinality, dataset, parentDivID)
    {
        this._name      = name;
        this._stage     = stage;
        this._panels    = {};
        this._target    = Utils.uuidv4();

        this._inputCardinality  = inputCardinality;
        this._outputCardinality = outputCardinality;
        this._dataset           = dataset;
        this._filteredIDs       = new Set();

        // Create div structure for this operator.
        let div         = document.createElement('div');
        div.id          = this._target;
        div.className   = 'operator';
        $("#" + (
            (typeof parentDivID == "undefined" || parentDivID == null) ? this._stage.target : parentDivID
        )).append(div);

        // Make class abstract.
        if (new.target === Operator) {
            throw new TypeError("Cannot construct Operator instances.");
        }
    }

    /**
     * Constructs all panels in this operator.
     */
    constructPanels()
    {
        throw new TypeError("Operator.constructPanels(): Abstract method must not be called.");
    }

    /**
     * (Re-)Renders all panels.
     */
    render()
    {
        for (let panelName in this._panels) {
            this._panels[panelName].render();
        }
    }

    /**
     * Resizes all panels.
     */
    resize()
    {
        for (let panelName in this._panels) {
            this._panels[panelName].render();
        }
    }

    /**
     * Updates current filtering by specifying which IDs are to be considered active.
     * Triggers filter() operations for its Panel instances. Propagates filter changes to its Stage.
     * @param embeddingIDs All active embedding IDs.
     */
    filter(embeddingIDs)
    {
        throw new TypeError("Operator.filter(): Abstract method must not be called.");
    }

    /**
     * Propagate changes after settings in corresponding SettingsPanel have been changed.
     * @param delta Resulting diff. from change in settings - can be a new dataset or just changes in configuration.
     * Note that Operator.propagateSettingsChanges() is agnostic towards the content - the associated panels/charts have
     * to deal with the data.
     * @param sourcePanelName Name of the panel acting as source.
     */
    propagateSettingsChanges(delta, sourcePanelName)
    {
        for (let key in this._panels) {
            if (this._panels[key]._name !== sourcePanelName)
                this._panels[key].processSettingsChange(delta);
        }
    }

    /**
     * Highlights data point in all charts in all panels in this operator and propagates event to stage.
     * @param id
     * @param source
     * @param propagate Determines whether highlight event should be propagated to stage.
     */
    highlight(id, source, propagate = false)
    {
        for (let key in this._panels) {
            if (this._panels[key]._name !== source)
                this._panels[key].highlight(id, source);
        }

        if (propagate)
            this._stage.highlight(id, this._name);
    }

    get name()
    {
        return this._name;
    }

    get panels()
    {
        return this._panels;
    }

    get dataset()
    {
        return this._dataset;
    }

    get stage()
    {
        return this._stage;
    }

    get target()
    {
        return this._target;
    }

    get filteredIDs()
    {
        return this._filteredIDs;
    }
}