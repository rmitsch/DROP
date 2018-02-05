/**
 * Holds all elements to be shown in a certain view.
 * Note that a dashboard corresponds to exactly one view, pop-ups/detail views correspond to another view.
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