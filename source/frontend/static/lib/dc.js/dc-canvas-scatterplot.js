/**
 * A scatter plot chart
 *
 * Examples:
 * - {@link http://dc-js.github.io/dc.js/examples/scatter.html Scatter Chart}
 * - {@link http://dc-js.github.io/dc.js/examples/multi-scatter.html Multi-Scatter Chart}
 * @class scatterPlot
 * @memberof dc
 * @mixes dc.coordinateGridMixin
 * @example
 * // create a scatter plot under #chart-container1 element using the default global chart group
 * var chart1 = dc.scatterPlot('#chart-container1');
 * // create a scatter plot under #chart-container2 element using chart group A
 * var chart2 = dc.scatterPlot('#chart-container2', 'chartGroupA');
 * // create a sub-chart under a composite parent chart
 * var chart3 = dc.scatterPlot(compositeChart);
 * @param {String|node|d3.selection} parent - Any valid
 * {@link https://github.com/d3/d3-3.x-api-reference/blob/master/Selections.md#selecting-elements d3 single selector} specifying
 * a dom block element such as a div; or a dom element or d3 selection.
 * @param {String} [chartGroup] - The name of the chart group this chart instance should be placed in.
 * Interaction with a chart will only trigger events and redraws within the chart's group.
 * @param {DRMetaDataset} dataset
 * @param {String} variantAttribute
 * @param {String} objective
 * @param {Boolean} useBinning Determines whether points should be plotted in (hexagonal) bins.
 * @returns {dc.scatterPlot}
 */


dc.scatterPlot = function (parent, chartGroup, dataset, variantAttribute, objective, useBinning = false) {
    var _chart = dc.coordinateGridMixin({});
    var _symbol = d3.svg.symbol();

    // Store references to dataset and variant attributes.
    _chart.dataset          = dataset;
    _chart.binCount         = dataset._binCountSSP;
    _chart.variantAttribute = variantAttribute;
    _chart.objective        = objective;
    _chart.useBinning       = useBinning;

    // Store last highlighted coordinates.
    _chart.lastHighlightedPosition = null;

    var _existenceAccessor = function (d) {
        return d.value;
    };

    var originalKeyAccessor = _chart.keyAccessor();
    _chart.keyAccessor(function (d) {
        return originalKeyAccessor(d)[0];
    });
    _chart.valueAccessor(function (d) {
        return originalKeyAccessor(d)[1];
    });
    _chart.colorAccessor(function () {
        return _chart._groupName;
    });

    _chart.title(function (d) {
        // this basically just counteracts the setting of its own key/value accessors
        // see https://github.com/dc-js/dc.js/issues/702
        return _chart.keyAccessor()(d) + ',' + _chart.valueAccessor()(d) + ': ' +
            _chart.existenceAccessor()(d);
    });

    var _locator = function (d) {
        return 'translate(' + _chart.x()(_chart.keyAccessor()(d)) + ',' +
            _chart.y()(_chart.valueAccessor()(d)) + ')';
    };

    var _highlightedSize = 7;
    var _symbolSize = 5;
    var _excludedSize = 3;
    var _excludedColor = null;
    var _excludedOpacity = 1.0;
    var _emptySize = 0;
    var _emptyOpacity = 0;
    var _nonemptyOpacity = 1;
    var _emptyColor = "#ccc";
    var _filtered = [];
    var _canvas = null;
    var _context = null;
    var _useCanvas = false;

    // Calculates element radius for canvas plot to be comparable to D3 area based symbol sizes
    function canvasElementSize(d, isFiltered) {
        if (!_existenceAccessor(d)) {
            return _excludedSize / Math.sqrt(Math.PI);
        } else if (isFiltered) {
            return _symbolSize / Math.sqrt(Math.PI);
        } else {
            return _excludedSize / Math.sqrt(Math.PI);
        }
    }

    function elementSize(d, i) {
        if (!_existenceAccessor(d)) {
            return Math.pow(_excludedSize, 2);
        } else if (_filtered[i]) {
            return Math.pow(_symbolSize, 2);
        } else {
            return Math.pow(_excludedSize, 2);
        }
    }

    _symbol.size(elementSize);

    dc.override(_chart, '_filter', function (filter) {
        if (!arguments.length) {
            return _chart.__filter();
        }

        return _chart.__filter(dc.filters.RangedTwoDimensionalFilter(filter));
    });

    _chart._resetSvgOld = _chart.resetSvg; // Copy original closure from base-mixin

    /**
     * Method that replaces original resetSvg and appropriately inserts canvas
     * element along with svg element and sets their CSS properties appropriately
     * so they are overlapped on top of each other.
     * Remove the chart's SVGElements from the dom and recreate the container SVGElement.
     * @method resetSvg
     * @memberof dc.scatterPlot
     * @instance
     * @see {@link https://developer.mozilla.org/en-US/docs/Web/API/SVGElement SVGElement}
     * @returns {SVGElement}
     */
    _chart.resetSvg = function () {
        if (!_useCanvas) {
            return _chart._resetSvgOld();
        } else {
            _chart._resetSvgOld(); // Perform original svgReset inherited from baseMixin
            _chart.select('canvas').remove(); // remove old canvas

            var svgSel = _chart.svg();
            var rootSel = _chart.root();

            // Set root node to relative positioning and svg to absolute
            rootSel.style('position', 'relative');
            svgSel.style('position', 'relative');

            // Check if SVG element already has any extra top/left CSS offsets
            var svgLeft = isNaN(parseInt(svgSel.style('left'), 10)) ? 0 : parseInt(svgSel.style('left'), 10);
            var svgTop = isNaN(parseInt(svgSel.style('top'), 10)) ? 0 : parseInt(svgSel.style('top'), 10);
            var width = _chart.effectiveWidth();
            var height = _chart.effectiveHeight();
            var margins = _chart.margins(); // {top: 10, right: 130, bottom: 42, left: 42}

            // Add the canvas element such that it perfectly overlaps the plot area of the scatter plot SVG
            var devicePixelRatio = window.devicePixelRatio || 1;
            _canvas = _chart.root().append('canvas')
                .attr('x', 0)
                .attr('y', 0)
                .attr('width', (width) * devicePixelRatio)
                .attr('height', (height) * devicePixelRatio)
                .style('width', width + 'px')
                .style('height', height + 'px')
                .style('position', 'absolute')
                .style('top', margins.top + svgTop + 'px')
                .style('left', margins.left + svgLeft + 'px')
                .style('pointer-events', 'none'); // Disable pointer events on canvas so SVG can capture brushing

            // Define canvas context and set clipping path
            _context = _canvas.node().getContext('2d');
            _context.scale(devicePixelRatio, devicePixelRatio);
            _context.rect(0, 0, width, height);
            _context.clip(); // Setup clipping path
            _context.imageSmoothingQuality = 'high';

            return _chart.svg(); // Respect original return param for _chart.resetSvg;
        }
    };

    /**
     * Set or get whether to use canvas backend for plotting scatterPlot. Note that the
     * canvas backend does not currently support
     * {@link dc.scatterPlot#customSymbol customSymbol} or
     * {@link dc.scatterPlot#symbol symbol} methods and is limited to always plotting
     * with filled circles. Symbols are drawn with
     * {@link dc.scatterPlot#symbolSize symbolSize} radius. By default, the SVG backend
     * is used when `useCanvas` is set to `false`.
     * @method useCanvas
     * @memberof dc.scatterPlot
     * @instance
     * @param {Boolean} [useCanvas=false]
     * @return {Boolean|d3.selection}
     */
    _chart.useCanvas = function (useCanvas) {
        if (!arguments.length) {
            return _useCanvas;
        }
        _useCanvas = useCanvas;
        return _chart;
    };

    /**
     * Set or get canvas element. You should usually only ever use the get method as
     * dc.js will handle canvas element generation.  Provides valid canvas only when
     * {@link dc.scatterPlot#useCanvas useCanvas} is set to `true`
     * @method canvas
     * @memberof dc.scatterPlot
     * @instance
     * @param {CanvasElement|d3.selection} [canvasElement]
     * @return {CanvasElement|d3.selection}
     */
    _chart.canvas = function (canvasElement) {
        if (!arguments.length) {
            return _canvas;
        }
        _canvas = canvasElement;
        return _chart;
    };

    /**
     * Get canvas 2D context. Provides valid context only when
     * {@link dc.scatterPlot#useCanvas useCanvas} is set to `true`
     * @method context
     * @memberof dc.scatterPlot
     * @instance
     * @return {CanvasContext}
     */
    _chart.context = function () {
        return _context;
    };

    /**
     * Plots data on canvas element. If argument provided, assumes legend is currently being highlighted and modifies
     * opacity/size of symbols accordingly
     *
     * CAUTION: This implementation was tuned for usage in https://github.com/rmitsch/drop.
     * There are hardcoded changes, meaning this version of dc.js can NOT be updated without losing the added/changed
     * functionality.
     *
     * @param legendHighlightDatum {Object} Datum provided to legendHighlight method
     */
    function plotOnCanvas(legendHighlightDatum) {
        var context = _chart.context();
        context.clearRect(0, 0, (context.canvas.width + 2) * 1, (context.canvas.height + 2) * 1);
        var data = _chart.data();

        // ---------------------------------------
        // 0. Get datapoints' coordinates.
        // ---------------------------------------

        const seriesMapping = _chart.dataset.seriesMappingByHyperparameter[_chart.variantAttribute];
        // Store association between data points' coordinates and their IDs (ID -> coordinates).
        let coordinatesToFilteredDataPoints = {};

        data.forEach(function (d, i) {
            let x       = _chart.x()(_chart.keyAccessor()(d));
            let y       = _chart.y()(_chart.valueAccessor()(d));
            let xRound  = Math.round(x);
            let yRound  = Math.round(y);

            // Update coordinate store.
            if (!(xRound in coordinatesToFilteredDataPoints))
                coordinatesToFilteredDataPoints[xRound] = {};
            if (!(yRound in coordinatesToFilteredDataPoints[xRound]))
                coordinatesToFilteredDataPoints[xRound][yRound] = {
                    ids: new Set(),
                    idsUnfiltered: new Set(),
                    attr_x: _chart.keyAccessor()(d),
                    attr_y: _chart.valueAccessor()(d),
                    seriesIDs: new Set(),
                    seriesIDsUnfiltered: new Set()
                };

            // Add datapoint to set of filtered/unfiltered coordinates.
            const isFiltered    = !_chart.filter() || _chart.filter().isFiltered([d.key[0], d.key[1]]);
            const setName       = isFiltered ? "ids" : "idsUnfiltered";
            const seriesSetName = isFiltered ? "seriesIDs" : "seriesIDsUnfiltered";
            let currCollection  = coordinatesToFilteredDataPoints[xRound][yRound];
            for (let datapoint of d.value.items) {
                currCollection[setName].add(datapoint.id);

                if (seriesMapping !== undefined)
                    currCollection[seriesSetName].add(seriesMapping.recordToSeriesMapping[datapoint.id]);
            }
        });

        // ---------------------------------------
        // 1. Draw lines.
        // ---------------------------------------

        // Remove '*' if this is a categorical attribute.
        let variantAttribute = _chart.variantAttribute;
        if (variantAttribute !== null) {
            variantAttribute = variantAttribute.includes("*") ?
                _chart.variantAttribute.substring(0, _chart.variantAttribute.length - 1) :
                _chart.variantAttribute;
        }

        // Only draw lines if series data for this attribute exists.
        // If not: Objective-objective pairing.
        if (_chart.dataset !== null &&
            _chart.dataset !== undefined &&
            !_chart.useBinning &&
            // Check if variant attribute is hyperparameter.
            _chart.dataset._metadata.hyperparameters.some(hp => hp.name === variantAttribute)
        ) {
            context.save();

            // Set global drawing options for lines.
            context.lineWidth   = _chart.effectiveHeight() / _chart.binCount / 4 / 2;
            context.strokeStyle = "#1f77b4";
            context.globalAlpha = 0.5; // .0275

            // Draw lines between points of a series.
            let extrema = plotLines(context, coordinatesToFilteredDataPoints);

            // Draw pareto frontiers.
            // Set global drawing options for lines.
            context.lineWidth   = 1;
            context.strokeStyle = "red";
            context.globalAlpha = 1;
            plotParetoFrontiers(context, extrema);

            context.restore();
        }

        // ---------------------------------------
        // 2. Draw points.
        // ---------------------------------------

        if (!_chart.useBinning)
            plotPointsOnCanvas(context, data, coordinatesToFilteredDataPoints);
    }

    /**
     * Draws lines between points in a series.
     * @param context
     * @param coordinatesToDataPoints
     * @returns {{}} extrema Dictionary holding extrema for all values of variant attribute.
     */
    function plotLines(context, coordinatesToDataPoints)
    {
        // Collect best- and worst-performing parametrizations for all variations of variant parameter.
        let extrema         = {};
        const seriesCount   = _chart.dataset.seriesMappingByHyperparameter[_chart.variantAttribute].seriesCount;
        const recordCount   = _chart.dataset._data.length;

        // -------------------------------------------------------------------------------------------------
        // Draw lines between points belonging to the same series.
        // Note: For quadratic interpolation see
        // https://stackoverflow.com/questions/7054272/how-to-draw-smooth-curve-through-n-points-using-javascript-html5-canvas.
        // -------------------------------------------------------------------------------------------------

        context.beginPath();
        context.lineWidth = 1;

        // Sort records in series (list of dictionaries) by attribute.
        // We don't know whether the attribute is numeric or not, so we don't use arithmetic operations to directly
        // determine the sort order. See
        // https://stackoverflow.com/questions/1129216/sort-array-of-objects-by-string-property-value-in-javascript.
        let sortFunction = function(a, b) {
            return (
                a[_chart.variantAttribute] > b[_chart.variantAttribute]) ? 1 :
                ((b[_chart.variantAttribute] > a[_chart.variantAttribute]) ? -1: 0
            );
        };
        let xCoords = Object.keys(coordinatesToDataPoints).sort(sortFunction);

        // Draw lines between each two adjacent grouping of points, i. e. two points with differing x/y coordinates.
        let lineVals = {};
        for (let xIndex1 = 0; xIndex1 < xCoords.length; xIndex1++) {
            let x1          = xCoords[xIndex1];
            let yCoords1    = Object.keys(coordinatesToDataPoints[x1]).sort(sortFunction);
            extrema[x1]     = {min: Infinity, max: -Infinity};

            for (let yIndex1 = 0; yIndex1 < yCoords1.length; yIndex1++) {
                let y1      = parseInt(yCoords1[yIndex1]);
                let set1    = coordinatesToDataPoints[x1][y1].seriesIDs;

                // Update extrema for Pareto frontier.
                if (set1.size > 0 && y1 < extrema[x1].min)
                    extrema[x1].min = y1;
                else if (set1.size > 0 && y1 > extrema[x1].max)
                    extrema[x1].max = y1;

                // Cycle through all possible combinations to points at next x value / bin.
                if (set1.size > 0 && xIndex1 + 1 < xCoords.length) {
                    let x2          = xCoords[xIndex1 + 1];
                    let yCoords2    = Object.keys(coordinatesToDataPoints[x2]).sort(sortFunction);

                    for (let yIndex2 = 0; yIndex2 < yCoords2.length; yIndex2 ++) {
                        let y2              = yCoords2[yIndex2];
                        let set2            = coordinatesToDataPoints[x2][y2].seriesIDs;
                        const intersection  = new Set([...set1].filter(x => set2.has(x)));

                        if (intersection.size > 0) {
                            let alpha = Math.ceil(intersection.size / seriesCount * 5) / 5;

                            if (!(alpha in lineVals))
                                lineVals[alpha] = [];
                            lineVals[alpha].push([
                                alpha, parseInt(x1), y1, parseInt(x2), parseInt(y2)
                            ]);
                        }
                    }
                }
            }
        }

        for (const alpha in lineVals) {
            context.globalAlpha = alpha;

            for (const lineValSet of lineVals[alpha]) {
                context.beginPath();
                context.moveTo(lineValSet[1], lineValSet[2]);
                context.lineTo(lineValSet[3], lineValSet[4]);
                context.stroke();
            }
        }
        context.restore();

        return extrema;
    }

    /**
     * Draws optimal and pessimal pareto frontiers.
     * Note: Does not use context.save() or context.restore().
     * @param context
     * @param extrema
     */
    function plotParetoFrontiers(context, extrema)
    {
        // Sort keys in extrema dictionary by their numerical sequence (numeric cast necessary before sort).
        let extremaKeys = Object.keys(extrema).sort();
        let sortedExtremaKeys = extremaKeys.sort(function (a, b) {
            return +a - +b;
        });

        // Go through points in sorted order, draw pareto-optimal and -pessimal (sic) frontiers.
        for (let i = 0; i < sortedExtremaKeys.length - 1; i++) {
            let key     = sortedExtremaKeys[i];
            let nextKey = sortedExtremaKeys[i + 1];

            // Draw line for pareto-optimal/-pessimal points.
            context.beginPath();
            context.moveTo(parseInt(key), extrema[key].min);
            context.lineTo(parseInt(nextKey), extrema[nextKey].min);
            context.stroke();

            // Draw lines for pareto-pessimal/-optimal points.
            context.beginPath();
            context.moveTo(parseInt(key), extrema[key].max);
            context.lineTo(parseInt(nextKey), extrema[nextKey].max);
            context.stroke();
        }
    }

    /**
     * Plots all points in scatterplot.
     * @param context
     * @param data
     * @param coordinatesToDataPoints
     */
    function plotPointsOnCanvas(context, data, coordinatesToDataPoints)
    {
        context.save();

        // ---------------------------------------------------------------------------------
        // 1. Gather data and separate in blocks with identical appearance/canvas settings
        // for performance optimization before plotting them.
        // ---------------------------------------------------------------------------------

        const pointRadius = _chart.variantAttribute == null && _chart.useBinning == null ?
            _chart.effectiveHeight() / _chart.binCount / 4 : _symbolSize;

        let dataPoints = {
            filtered: {
                color: _chart.getColor(data[0]),
                radius: pointRadius, // canvasElementSize(data[0], true)
                opacity: _nonemptyOpacity,
                coordinates: []
            },
            notFiltered: {
                color: _emptyColor,
                radius: pointRadius, // canvasElementSize(data[0], true)
                opacity: _chart.excludedOpacity(),
                coordinates: []
            }
        };

        for (let x in coordinatesToDataPoints) {
            for (let y in coordinatesToDataPoints[x]) {
                if (coordinatesToDataPoints[x][y].ids.size > 0)
                    dataPoints.filtered.coordinates.push([parseInt(x), parseInt(y)]);
                if (coordinatesToDataPoints[x][y].idsUnfiltered.size > 0)
                    dataPoints.notFiltered.coordinates.push([parseInt(x), parseInt(y)]);
            }
        }

        // ---------------------------------------------------------------------------------
        // 2. Plot points.
        // ---------------------------------------------------------------------------------

        // Plot all filtered points.
        // a. Configure canvas context.
        context.globalAlpha = dataPoints.filtered.opacity;
        context.fillStyle   = dataPoints.filtered.color;
        context.beginPath();
        // b. Plot points.
        for (let coordinate of dataPoints.filtered.coordinates) {
            context.moveTo(coordinate[0], coordinate[1]);
            context.arc(
                coordinate[0],
                coordinate[1],
                dataPoints.filtered.radius,
                0,
                2 * Math.PI,
                true
            );
        }
        context.fill();
        context.save();

        // Plot all unfiltered points.
        // b. Configure canvas context.
        context.globalAlpha = dataPoints.notFiltered.opacity;
        context.fillStyle   = dataPoints.notFiltered.color;
        context.beginPath();
        // b. Plot points.
        for (let coordinate of dataPoints.notFiltered.coordinates) {
            context.moveTo(coordinate[0], coordinate[1]);
            context.arc(
                coordinate[0],
                coordinate[1],
                dataPoints.notFiltered.radius,
                0,
                2 * Math.PI,
                true
            );
        }
        context.fill();
        context.save();

        context.restore();
    }

    _chart.highlight = function(id)
    {
        // Delete old highlighting.
        _chart.chartBodyG()
            .selectAll("*")
            .remove();

        // If ID is null: Don't highlight new point.
        if (id !== null) {
            let recordToHighlight   = null;
            let indexToHighlight    = null;
            let data                = _chart.data();

            // Get point to highlight.
            for (let i = 0; i < data.length && recordToHighlight === null; i++) {
                for (let datapoint of data[i].value.items) {
                    if (datapoint.id === id) {
                        indexToHighlight = i;
                        recordToHighlight = [datapoint];
                        recordToHighlight[0].coordinates = {
                            x: _chart.x()(_chart.keyAccessor()(data[i])),
                            y: _chart.y()(_chart.valueAccessor()(data[i])),
                        };
                    }
                }
            }

            // Exit if point not found (shouldn't happen).
            if (recordToHighlight === null)
                throw "dc-canvas-scatterplot.highlight(): Record ID not found.";

            // Draw circles for highlighted points.
            let circles = _chart.chartBodyG()
                .selectAll("circle")
                .data(recordToHighlight)
                .enter()
                .append("circle");

            // Draw circles.
            let useLastHighlightedPosition = _chart.lastHighlightedPosition !== null;
            circles
                .attr("opacity", 1)
                .attr("r", 5)
                .attr("cx", function (d) {
                    return useLastHighlightedPosition ? _chart.lastHighlightedPosition.x : d.coordinates.x;
                })
                .attr("cy", function (d) {
                    return useLastHighlightedPosition ? _chart.lastHighlightedPosition.y : d.coordinates.y;
                })
                .style("fill", "red");
            dc.transition(circles, 100, _chart.transitionDelay())
                .attr("cx", function (d) { return d.coordinates.x; })
                .attr("cy", function (d) { return d.coordinates.y; })
                .attr("opacity", 1);

            // Store last highlighted position.
            _chart.lastHighlightedPosition = recordToHighlight[0].coordinates;
        }
    };

    _chart.plotData = function () {
        if (_useCanvas) {
            plotOnCanvas();
        } else {
            var symbols = _chart.chartBodyG().selectAll('path.symbol')
                .data(_chart.data());

            symbols
                .enter()
                .append('path')
                .attr('class', 'symbol')
                .attr('opacity', 0)
                .attr('fill', _chart.getColor)
                .attr('transform', _locator);

            symbols.call(renderTitles, _chart.data());

            symbols.each(function (d, i) {
                _filtered[i] = !_chart.filter() || _chart.filter().isFiltered([d.key[0], d.key[1]]);
            });

            dc.transition(symbols, _chart.transitionDuration(), _chart.transitionDelay())
                .attr('opacity', function (d, i) {
                    if (!_existenceAccessor(d)) {
                        return _chart.excludedOpacity();
                    } else if (_filtered[i]) {
                        return _nonemptyOpacity;
                    } else {
                        return _chart.excludedOpacity();
                    }
                })
                .attr('fill', function (d, i) {
                    if (_emptyColor && !_existenceAccessor(d)) {
                        return _emptyColor;
                    } else if (!_filtered[i]) {
                        return _chart.excludedColor();
                    } else {
                        return _chart.getColor(d);
                    }
                })
                .attr('transform', _locator)
                .attr('d', _symbol);

            dc.transition(symbols.exit(), _chart.transitionDuration(), _chart.transitionDelay())
                .attr('opacity', 0).remove();
        }
    };

    function renderTitles(symbol, d) {
        if (_chart.renderTitle()) {
            symbol.selectAll('title').remove();
            symbol.append('title').text(function (d) {
                return _chart.title()(d);
            });
        }
    }

    /**
     * Get or set the existence accessor.  If a point exists, it is drawn with
     * {@link dc.scatterPlot#symbolSize symbolSize} radius and
     * opacity 1; if it does not exist, it is drawn with
     * {@link dc.scatterPlot#emptySize emptySize} radius and opacity 0. By default,
     * the existence accessor checks if the reduced value is truthy.
     * @method existenceAccessor
     * @memberof dc.scatterPlot
     * @instance
     * @see {@link dc.scatterPlot#symbolSize symbolSize}
     * @see {@link dc.scatterPlot#emptySize emptySize}
     * @example
     * // default accessor
     * chart.existenceAccessor(function (d) { return d.value; });
     * @param {Function} [accessor]
     * @returns {Function|dc.scatterPlot}
     */
    _chart.existenceAccessor = function (accessor) {
        if (!arguments.length) {
            return _existenceAccessor;
        }
        _existenceAccessor = accessor;
        return this;
    };

    /**
     * Get or set the symbol type used for each point. By default the symbol is a circle.
     * Type can be a constant or an accessor.
     * @method symbol
     * @memberof dc.scatterPlot
     * @instance
     * @see {@link https://github.com/d3/d3-3.x-api-reference/blob/master/SVG-Shapes.md#symbol_type d3.svg.symbol.type}
     * @example
     * // Circle type
     * chart.symbol('circle');
     * // Square type
     * chart.symbol('square');
     * @param {String|Function} [type='circle']
     * @returns {String|Function|dc.scatterPlot}
     */
    _chart.symbol = function (type) {
        if (!arguments.length) {
            return _symbol.type();
        }
        _symbol.type(type);
        return _chart;
    };

    /**
     * Get or set the symbol generator. By default `dc.scatterPlot` will use
     * {@link https://github.com/d3/d3-3.x-api-reference/blob/master/SVG-Shapes.md#symbol d3.svg.symbol()}
     * to generate symbols. `dc.scatterPlot` will set the
     * {@link https://github.com/d3/d3-3.x-api-reference/blob/master/SVG-Shapes.md#symbol_size size accessor}
     * on the symbol generator.
     * @method customSymbol
     * @memberof dc.scatterPlot
     * @instance
     * @see {@link https://github.com/d3/d3-3.x-api-reference/blob/master/SVG-Shapes.md#symbol d3.svg.symbol}
     * @see {@link https://stackoverflow.com/questions/25332120/create-additional-d3-js-symbols Create additional D3.js symbols}
     * @param {String|Function} [customSymbol=d3.svg.symbol()]
     * @returns {String|Function|dc.scatterPlot}
     */
    _chart.customSymbol = function (customSymbol) {
        if (!arguments.length) {
            return _symbol;
        }
        _symbol = customSymbol;
        _symbol.size(elementSize);
        return _chart;
    };

    /**
     * Set or get radius for symbols.
     * @method symbolSize
     * @memberof dc.scatterPlot
     * @instance
     * @see {@link https://github.com/d3/d3-3.x-api-reference/blob/master/SVG-Shapes.md#symbol_size d3.svg.symbol.size}
     * @param {Number} [symbolSize=3]
     * @returns {Number|dc.scatterPlot}
     */
    _chart.symbolSize = function (symbolSize) {
        if (!arguments.length) {
            return _symbolSize;
        }
        _symbolSize = symbolSize;
        return _chart;
    };

    /**
     * Set or get radius for highlighted symbols.
     * @method highlightedSize
     * @memberof dc.scatterPlot
     * @instance
     * @see {@link https://github.com/d3/d3-3.x-api-reference/blob/master/SVG-Shapes.md#symbol_size d3.svg.symbol.size}
     * @param {Number} [highlightedSize=5]
     * @returns {Number|dc.scatterPlot}
     */
    _chart.highlightedSize = function (highlightedSize) {
        if (!arguments.length) {
            return _highlightedSize;
        }
        _highlightedSize = highlightedSize;
        return _chart;
    };

    /**
     * Set or get size for symbols excluded from this chart's filter. If null, no
     * special size is applied for symbols based on their filter status.
     * @method excludedSize
     * @memberof dc.scatterPlot
     * @instance
     * @see {@link https://github.com/d3/d3-3.x-api-reference/blob/master/SVG-Shapes.md#symbol_size d3.svg.symbol.size}
     * @param {Number} [excludedSize=null]
     * @returns {Number|dc.scatterPlot}
     */
    _chart.excludedSize = function (excludedSize) {
        if (!arguments.length) {
            return _excludedSize;
        }
        _excludedSize = excludedSize;
        return _chart;
    };

    /**
     * Set or get color for symbols excluded from this chart's filter. If null, no
     * special color is applied for symbols based on their filter status.
     * @method excludedColor
     * @memberof dc.scatterPlot
     * @instance
     * @param {Number} [excludedColor=null]
     * @returns {Number|dc.scatterPlot}
     */
    _chart.excludedColor = function (excludedColor) {
        if (!arguments.length) {
            return _excludedColor;
        }
        _excludedColor = excludedColor;
        return _chart;
    };

    /**
     * Set or get opacity for symbols excluded from this chart's filter.
     * @method excludedOpacity
     * @memberof dc.scatterPlot
     * @instance
     * @param {Number} [excludedOpacity=1.0]
     * @returns {Number|dc.scatterPlot}
     */
    _chart.excludedOpacity = function (excludedOpacity) {
        if (!arguments.length) {
            return _excludedOpacity;
        }
        _excludedOpacity = excludedOpacity;
        return _chart;
    };

    /**
     * Set or get radius for symbols when the group is empty.
     * @method emptySize
     * @memberof dc.scatterPlot
     * @instance
     * @see {@link https://github.com/d3/d3-3.x-api-reference/blob/master/SVG-Shapes.md#symbol_size d3.svg.symbol.size}
     * @param {Number} [emptySize=0]
     * @returns {Number|dc.scatterPlot}
     */
    _chart.hiddenSize = _chart.emptySize = function (emptySize) {
        if (!arguments.length) {
            return _emptySize;
        }
        _emptySize = emptySize;
        return _chart;
    };

    /**
     * Set or get color for symbols when the group is empty. If null, just use the
     * {@link dc.colorMixin#colors colorMixin.colors} color scale zero value.
     * @name emptyColor
     * @memberof dc.scatterPlot
     * @instance
     * @param {String} [emptyColor=null]
     * @return {String}
     * @return {dc.scatterPlot}/
     */
    _chart.emptyColor = function (emptyColor) {
        if (!arguments.length) {
            return _emptyColor;
        }
        _emptyColor = emptyColor;
        return _chart;
    };

    /**
     * Set or get opacity for symbols when the group is empty.
     * @name emptyOpacity
     * @memberof dc.scatterPlot
     * @instance
     * @param {Number} [emptyOpacity=0]
     * @return {Number}
     * @return {dc.scatterPlot}
     */
    _chart.emptyOpacity = function (emptyOpacity) {
        if (!arguments.length) {
            return _emptyOpacity;
        }
        _emptyOpacity = emptyOpacity;
        return _chart;
    };

    /**
     * Set or get opacity for symbols when the group is not empty.
     * @name nonemptyOpacity
     * @memberof dc.scatterPlot
     * @instance
     * @param {Number} [nonemptyOpacity=1]
     * @return {Number}
     * @return {dc.scatterPlot}
     */
    _chart.nonemptyOpacity = function (nonemptyOpacity) {
        if (!arguments.length) {
            return _emptyOpacity;
        }
        _nonemptyOpacity = nonemptyOpacity;
        return _chart;
    };

    _chart.legendables = function () {
        return [{chart: _chart, name: _chart._groupName, color: _chart.getColor()}];
    };

    _chart.legendHighlight = function (d) {
        if (_useCanvas) {
            plotOnCanvas(d); // Supply legend datum to plotOnCanvas
        } else {
            resizeSymbolsWhere(function (symbol) {
                return symbol.attr('fill') === d.color;
            }, _highlightedSize);
            _chart.chartBodyG().selectAll('.chart-body path.symbol').filter(function () {
                return d3.select(this).attr('fill') !== d.color;
            }).classed('fadeout', true);
        }
    };

    _chart.legendReset = function (d) {
        if (_useCanvas) {
            plotOnCanvas();
        } else {
            resizeSymbolsWhere(function (symbol) {
                return symbol.attr('fill') === d.color;
            }, _symbolSize);
            _chart.chartBodyG().selectAll('.chart-body path.symbol').filter(function () {
                return d3.select(this).attr('fill') !== d.color;
            }).classed('fadeout', false);
        }
    };

    function resizeSymbolsWhere(condition, size) {
        var symbols = _chart.chartBodyG().selectAll('.chart-body path.symbol').filter(function () {
            return condition(d3.select(this));
        });
        var oldSize = _symbol.size();
        _symbol.size(Math.pow(size, 2));
        dc.transition(symbols, _chart.transitionDuration(), _chart.transitionDelay()).attr('d', _symbol);
        _symbol.size(oldSize);
    }

    _chart.setHandlePaths = function () {
        // no handle paths for poly-brushes
    };

    _chart.extendBrush = function () {
        var extent = _chart.brush().extent();
        if (_chart.round()) {
            extent[0] = extent[0].map(_chart.round());
            extent[1] = extent[1].map(_chart.round());

            _chart.g().select('.brush')
                .call(_chart.brush().extent(extent));
        }
        return extent;
    };

    _chart.brushIsEmpty = function (extent) {
        return _chart.brush().empty() || !extent || extent[0][0] >= extent[1][0] || extent[0][1] >= extent[1][1];
    };

    _chart._brushing = function () {
        var extent = _chart.extendBrush();

        _chart.redrawBrush(_chart.g());

        if (_chart.brushIsEmpty(extent)) {
            dc.events.trigger(function () {
                _chart.filter(null);
                _chart.redrawGroup();
            });

        } else {
            var ranged2DFilter = dc.filters.RangedTwoDimensionalFilter(extent);
            dc.events.trigger(function () {
                _chart.filter(null);
                _chart.filter(ranged2DFilter);
                _chart.redrawGroup();
            }, dc.constants.EVENT_DELAY);

        }
    };

    _chart.setBrushY = function (gBrush) {
        gBrush.call(_chart.brush().y(_chart.y()));
    };

    return _chart.anchor(parent, chartGroup);
};
