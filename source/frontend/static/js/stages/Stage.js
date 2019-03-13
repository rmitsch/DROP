/**
 * Holds all elements to be shown in a certain view.
 * Note that a dashboard corresponds to exactly one view, pop-ups/detail views correspond to another view.
 * Hence: One Stage represents one concept/one idea for a finished product.
 */
export default class Stage
{
    /**
     *
     * @param name
     * @param target ID of container div.
     * @param datasets Dictionary of instances of dataset class.
     */
    constructor(name, target, datasets)
    {
        this._name      = name;
        this._target    = target;
        this._datasets  = datasets;
        this._operators = {};


        // Store info about key usage.
        this._shiftDown                 = false;
        this._ctrlDown                  = false;
        this._extendedSelectionEnabled  = false;
        this._keyEventCallbacks = [];
        this._initKeyDownListener();

        // Make class abstract.
        if (new.target === Stage) {
            throw new TypeError("Cannot construct Stage instances.");
        }
    }

    /**
     * Construct panels.
     */
    constructOperators()
    {
        throw new TypeError("Stage.constructOperators(): Cannot execute abstract method.");
    }

    /**
     * Updates current filtering by specifying which IDs are to be considered active.
     * @param source ID of operator triggering change.
     * @param embeddingIDs All active embedding IDs.
     */
    filter(source, embeddingIDs)
    {
        throw new TypeError("Stage.filter(source, embeddingIDs): Cannot execute abstract method.");
    }

    /**
     * Highlights datapoint in all operators.
     * @param id ID of embedding record to highlight.
     * @param source Name of operator instance.
     */
    highlight(id, source)
    {
        for (let key in this._operators) {
            if (this._operators[key]._name !== source)
                this._operators[key].highlight(id, source);
        }
    }

    /**
     * Listens to document-level key events.
     * @private
     */
    _initKeyDownListener()
    {
        let scope = this;

        function handleKeyStateChanges(e)
        {
            scope._shiftDown    = e.shiftKey;
            scope._ctrlDown     = e.ctrlKey;

            // Check activation status of extended selection mode.
            if (e.key === "h" && e.type === "keyup") {
                scope._extendedSelectionEnabled = !scope._extendedSelectionEnabled;
                const hoverInfo = $("#hovermode-info");
                hoverInfo.stop(true);
                hoverInfo.html((scope._extendedSelectionEnabled ? "Enabled" : "Disabled") + " extended selection mode.")
                hoverInfo.fadeIn(1000).delay(4000).fadeOut(1000);
            }

            for (let listener of scope._keyEventCallbacks)
                listener.callback(listener.instance, e);
        }

        document.addEventListener("keydown", handleKeyStateChanges, true);
        document.addEventListener("keyup", handleKeyStateChanges, true);
    }

    addKeyEventListener(instance, callback)
    {
        this._keyEventCallbacks.push({instance: instance, callback: callback});
    }

    get name()
    {
        return this._name;
    }

    get datasets()
    {
        return this._datasets;
    }

    get operators()
    {
        return this._operators;
    }

    get target()
    {
        return this._target;
    }
}