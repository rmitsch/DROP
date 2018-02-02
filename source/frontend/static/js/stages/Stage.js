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
     * @param data Array of objects (JSON/array/dict/...) holding data to display. Note: Length of array defines number
     * of panels (one dataset per panel) and has to be equal with length of objects in metadata.
     * @param metadata Array of JSON objects holding metadata. Note: Length of array has to be equal with length of
     * data.
     */
    constructor(name, target, data, metadata)
    {
        this._name = name;
        this._target = target;
        this._data = data;
        this._metadata = metadata;
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

    get data()
    {
        return this._data;
    }

    get metadata()
    {
        return this._metadata;
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