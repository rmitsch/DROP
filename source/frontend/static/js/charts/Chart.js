import Utils from "../Utils.js";

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
     * @param dataset
     * @param style Various style settings (chart width/height, colors, ...). Arbitrary format, has to be parsed indivdually
     * by concrete classes.
     * @param targetDivID
     */
    constructor(name, panel, attributes, dataset, style, targetDivID)
    {
        this._name          = name;
        this._panel         = panel;
        this._attributes    = attributes;
        this._dataset       = dataset;
        this._style         = style;
        this._target        = targetDivID == null ? Utils.uuidv4() : targetDivID;

        // Create div structure for this chart.
        if (targetDivID == null) {
           let div = document.createElement('div');
            div.id = this._target;
            div.className = 'chart';
            $("#" + this._panel.target).append(div);
        }

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

    get dataset()
    {
        return this._dataset;
    }
}