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
     * @param parentDivID
     */
    constructor(name, panel, attributes, dataset, style, parentDivID)
    {
        this._name              = name;
        this._panel             = panel;
        this._attributes        = attributes;
        this._dataset           = dataset;
        this._style             = style;
        this._target            = Utils.uuidv4();
        // Store size of panel at time of last render.
        this._lastPanelSize  = {width: 0, height: 0};

        // Create div structure for this chart.
        let div = document.createElement('div');
        div.id = this._target;
        div.className = 'chart';
        $("#" + (typeof parentDivID == "undefined" ? this._panel.target : parentDivID)).append(div);

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
     * Resize chart.
     * Note: Usually not necessary due to usage of dc.renderAll() and automatic crossfilter updates.
     */
    resize()
    {
        throw new TypeError("Chart.resize(): Abstract method must not be called.");
    }


    /**
     * Constructs and defines styling and behaviour of crossfilter's chart object.
     */
    constructCFChart()
    {
        throw new TypeError("Chart.constructCFChart(): Abstract method must not be called.");
    }

    /**
     * Highlights representation of record with this ID in this chart.
     * @param id
     * @param source
     */
    highlight(id, source)
    {
        throw new TypeError("Chart.highlight(): Abstract method must not be called.");
    }

    /**
     * Propagate filter changes to stage.
     * @param instance
     * @param key
     * @returns Set<int> Filtered IDs.
     */
    propagateFilterChange(instance, key)
    {
        let dimensions      = instance._dataset._cf_dimensions;
        let operator        = instance._panel._operator;
        let embeddingIDs    = new Set();

        dimensions[key].top(Infinity).forEach(record => embeddingIDs.add(record.id));

        if (!(Utils.compareSets(embeddingIDs, operator._filteredIDs))) {
            operator._filteredIDs = embeddingIDs;
            operator._stage.filter(operator._name, embeddingIDs);
        }

        return embeddingIDs;
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