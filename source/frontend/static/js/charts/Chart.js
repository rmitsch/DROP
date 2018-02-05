import uuidv4 from "../utils.js";

/**
 * Abstract base class for individual charts.
 * One instance of Chart is associated with exactly one instance of Panel.
 */
export default class Chart
{
    /**
     *
     * @param name
     * @param panel
     * @param attributes Attributes that are to be considered in this chart (how exactly is up to the implementation of
     * the relevant subclass(es)).
     * @param crossfilter
     */
    constructor(name, panel, attributes, crossfilter)
    {
        this._name          = name;
        this._panel         = panel;
        this._attributes    = attributes;
        this._crossfilter   = crossfilter;
        this._target        = uuidv4();

        // Define variables relevant for crossfilter.
        this._cf_dimensions = {};
        this._cf_extrema    = {};
        this._cf_groups     = {};
        this._cf_chart      = null;

        // Create div structure for this chart.
        let div         = document.createElement('div');
        div.id          = this._target;
        div.className   = 'chart';
        $("#" + this._panel.target).append(div);

        // Make class abstract.
        if (new.target === Chart) {
            throw new TypeError("Cannot construct Chart instances.");
        }
    }

    /**
     * (Re-)Render chart.
     * Note: Usually not necessary due to usage of dc.renderAll() and automatic crossfilter updates.
     */
    render()
    {
        throw new TypeError("Chart.render(): Abstract method must not be called.");
    }


    /**
     * Constructs and defines styling and behaviour of crossfilter's chart object.
     */
    constructCFChart()
    {
        throw new TypeError("Chart.constructCFChart(): Abstract method must not be called.");
    }

    get name()
    {
        return this._name;
    }

    get panel()
    {
        return this._panel;
    }

    get attributes()
    {
        return this._attributes;
    }
}