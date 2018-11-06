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