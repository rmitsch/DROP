/**
 *
 */
class Operator
{
    constructor(name, type)
    {
        this._name = name;
        this._panels = [];

        // Make class abstract.
        if (new.target === Chart) {
            throw new TypeError("Cannot construct Operator instances.");
        }
    }

    get name()
    {
        return this._name;
    }
}