/**
 * A panel holds exactly one chart plus optional controls.
 * Panel is linked with exactly one operator.
 */
export default class Panel
{
    constructor(name)
    {
        this._name = name;
        this._charts = [];

        // Make class abstract.
        if (new.target === Panel) {
            throw new TypeError("Cannot construct Panel instances.");
        }
    }

    get name()
    {
        return this._name;
    }
}