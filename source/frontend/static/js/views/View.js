/**
 * Holds all elements to be shown in a certain view.
 * Note that a dashboard corresponds to exactly one view, pop-ups/detail views correspond to another view.
 */


export default class View
{
    /**
     *
     * @param name
     * @param data Array of JSON objects holding data to display.
     * @param metadata JSON object holding metadata.
     */
    constructor(name, data, metadata)
    {
        this._name = name;
        this._operators = [];

        // Make class abstract.
        if (new.target === View) {
            throw new TypeError("Cannot construct Chart instances.");
        }
        this._data = data;
        this._metadata = metadata;
    }

    get name()
    {
        return this._name;
    }

    get data() {
        return this._data;
    }

    get metadata() {
        return this._metadata;
    }

    get operators() {
        return this._operators;
    }
}