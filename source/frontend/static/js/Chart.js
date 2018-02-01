/**
 *
 */
class Chart
{
    constructor(name, type)
    {
        this.name = name;
        
        // Make class abstract.
        if (new.target === Panel) {
            throw new TypeError("Cannot construct Panel instances.");
        }
    }

    get name()
    {
        return this.name;
    }
}