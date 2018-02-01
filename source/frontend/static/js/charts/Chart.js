/**
 *
 */
class Chart
{
    constructor(name, type)
    {
        this._name = name;
        
        // Make class abstract.
        if (new.target === Chart) {
            throw new TypeError("Cannot construct Chart instances.");
        }
    }

    get name()
    {
        return this._name;
    }
}