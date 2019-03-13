import Histogram from "./Histogram.js"
import Utils from "../Utils.js";

/**
 * Creates numerical histogram.
 */
export default class NumericalHistogram extends Histogram
{
    /**
     *
     * @param name
     * @param panel
     * @param attributes
     * @param dataset
     * @param style
     * @param parentDivID
     */
    constructor(name, panel, attributes, dataset, style, parentDivID)
    {
        super(name, panel, attributes, dataset, style, parentDivID);

        // For adaptions to click propagation handling.
        this._suppressNativeClickPropagation = true;
        this._panel._operator._stage.addKeyEventListener(this, NumericalHistogram.processKeyEvent);
    }

    render()
    {
        this._cf_chart.render();
    }

    constructCFChart()
    {
        // Use operator's target ID as group name.
        this._cf_chart  = dc.barChart("#" + this._target, this._panel._operator._target);
        let d3Chart     = d3.select("#" + this._target);
        this._suppressNativeClickPropagation =
            this._suppressNativeClickPropagation === undefined ? true : this._suppressNativeClickPropagation;

        // Create shorthand references.
        let instance    = this;
        let extrema     = this._dataset._cf_extrema;
        let intervals   = this._dataset._cf_intervals;
        let dimensions  = this._dataset._cf_dimensions;
        let key         = this._axes_attributes.x + "#histogram";
        // Use padding so that first/last bar are not cut off in chart.
        let dataPadding = intervals[this._axes_attributes.x] * this._style.paddingFactor;

        // Configure chart.
        this._cf_chart
            .height(instance._style.height)
            .width(instance._style.width)
            .valueAccessor( function(d) { return  d.value.count; } )
            .elasticY(false)
            .x(d3.scale.linear().domain([
                extrema[instance._axes_attributes.x].min - dataPadding,
                extrema[instance._axes_attributes.x].max + dataPadding
            ]))
            .y(d3.scale.linear().domain([0, extrema[key].max]))
            .brushOn(true)
            // Filter on end of brushing action, not meanwhile (performance suffers otherwise).
            .filterOnBrushEnd(true)
            .dimension(dimensions[key])
            .group(this._dataset.cf_groups[key])
            .renderHorizontalGridLines(true)
            .margins({top: 0, right: 10, bottom: 16, left: 25})
            .gap(1)
            // Call cross-operator filter method on stage instance after filter event.
            .on("filtered", event => {
                this._filteredIDs = this.propagateFilterChange(this, key);
                this._panel._operator.filter(this._filteredIDs);
                this._panel._operator.render();
            })
            .on('renderlet', function(chart) {
                // Suppress propagation of mouse click events.
                if (instance._suppressNativeClickPropagation) {
                    d3Chart.select(".brush").style({"pointer-events": "none"});

                    chart.selectAll('rect.bar')
                        .on('mouseover', function(d) {
                            d3.select(this).attr('fill', 'rgb(255, 0, 0)');
                        })
                        .on('mouseout', function(d) {
                            d3.select(this).attr('fill', 'rgb(31, 119, 180)');
                        });
                }
            });

        // Intercept mousedown so we can have both brush and mouseover.
        this._updateMouseDownListener();

        // Set number of ticks.
        this._cf_chart.yAxis().ticks(instance._style.numberOfTicks.y);
        this._cf_chart.xAxis().ticks(instance._style.numberOfTicks.x);

        // Update bin width.
        const binWidth = this._dataset._cf_intervals[this._axes_attributes.x] / this._dataset._binCount;
        this._cf_chart.xUnits(dc.units.fp.precision(binWidth));
    }

    /**
     * Updates mousedown listener w.r.t. value of _suppressNativeClickPropagation.
     * @private
     */
    _updateMouseDownListener()
    {
        let d3Chart = d3.select("#" + this._target);

        d3Chart.on('mousedown', this._suppressNativeClickPropagation ?
            function() {
                let brush               = d3Chart.select(".brush");
                let new_click_event     = new Event('mousedown');
                new_click_event.pageX   = d3.event.pageX;
                new_click_event.clientX = d3.event.clientX;
                new_click_event.pageY   = d3.event.pageY;
                new_click_event.clientY = d3.event.clientY;

                brush.node().dispatchEvent(new_click_event);
            } : null
        );
    }

    /**
     * Process global key events.
     * @param instance
     * @param keyEvent
     */
    static processKeyEvent(instance, keyEvent)
    {
        if (keyEvent.key === "h" && keyEvent.type === "keyup") {
            instance._suppressNativeClickPropagation = !instance._suppressNativeClickPropagation;
            instance._updateMouseDownListener();
            d3.select("#" + instance._target).select(".brush").style(
                {"pointer-events": instance._suppressNativeClickPropagation ? "none" : "all"}
            );
        }
    }

    highlight(id, source)
    {
        if (source !== this._name) {
            if (id !== null) {
                this._cf_chart.selectAll('rect.bar').each(function(d) {
                    if (d.data.value.ids.has(id))
                        d3.select(this).attr("fill", "red");
                });
            }

            // Reset all bars to default color.
            else {
                this._cf_chart.selectAll('rect.bar').each(function(d){
                    d3.select(this).attr("fill", "#1f77b4");
                });
            }
        }
    }
}