import uuidv4 from "../utils.js";

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
     * @param data
     * @param metadata
     */
    constructor(name, stage, inputCardinality, outputCardinality, data, metadata)
    {
        this._name = name;
        this._stage = stage;
        this._panels = {};
        this._target = uuidv4();

        this._inputCardinality = inputCardinality;
        this._outputCardinality = outputCardinality;
        this._data = data;
        this._metadata = metadata;


        // Create div structure for this operator.
        let div = document.createElement('div');
        div.id = this._target;
        div.className = 'operator filter-reduce-operator';
        $("#" + this._stage.target).append(div);

        // Make class abstract.
        if (new.target === Operator) {
            throw new TypeError("Cannot construct Operator instances.");
        }
    }

    constructPanels()
    {
        throw new TypeError("Operator.constructPanels: Abstract method must not be called.");
    }

    get name()
    {
        return this._name;
    }

    get panels()
    {
        return this._panels;
    }

    get inputCardinality()
    {
        return this._inputCardinality;
    }

    get outputCardinality()
    {
        return this._outputCardinality;
    }

    get data()
    {
        return this._data;
    }

    get metadata()
    {
        return this._metadata;
    }

    get stage()
    {
        return this._stage;
    }

    get target()
    {
        return this._target;
    }
}