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
 * @param {Boolean} useBinning Determines whether points should be plotted in (hexagonal) bins.
 * @returns {dc.scatterPlot}
 */
dc.scatterPlot = function (parent, chartGroup, dataset, variantAttribute, useBinning = false) {
    var _chart = dc.coordinateGridMixin({});
    var _symbol = d3.svg.symbol();

    // Store references to dataset and variant attributes.
    _chart.dataset = dataset;
    _chart.variantAttribute = variantAttribute;
    _chart.useBinning = useBinning;

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

        // Store association between data points' coordinates and their IDs (ID -> coordinates).
        let dataPointCoordinates = {};
        data.forEach(function (d, i) {
            // Store coordinates.
            let x = _chart.x()(_chart.keyAccessor()(d));
            let y = _chart.y()(_chart.valueAccessor()(d));

            // Only add datapoint to set of coordinates to draw if it's a filtered one.
            let isFiltered = !_chart.filter() || _chart.filter().isFiltered([d.key[0], d.key[1]]);
            if (isFiltered) {
                for (let datapoint of d.value.items) {
                    // Round coordinates to increase performance (avoid sub-pixel coordinates).
                    dataPointCoordinates[datapoint.id] = [Math.round(x), Math.round(y)];
                }
            }
        });

        // ---------------------------------------
        // 1. Draw lines.
        // ---------------------------------------

        // Only draw lines if series data for this attribute exists.
        // If not: Probably objective-objective pairing - connection has to be determined on-the-fly/on-demand (not
        // implemented yet).

        if (_chart.dataset !== null &&
            _chart.dataset !== undefined &&
            _chart.variantAttribute in _chart.dataset._seriesMappingByHyperparameter
        ) {
            context.save();

            // Set global drawing options for lines.
            context.lineWidth   = 1;
            context.strokeStyle = "#1f77b4";
            context.globalAlpha = 0.0275;

            // Draw lines between points of a series.
            let extrema = plotLines(context, dataPointCoordinates);

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
            plotPointsOnCanvas(context, data, dataPointCoordinates);
    }

    /**
     * Draws lines between points in a series.
     * @param context
     * @param dataPointCoordinates
     * @returns {{}} extrema Dictionary holding extrema for all values of variant attribute.
     */
    function plotLines(context, dataPointCoordinates)
    {
        // Fetch mapping from dataset ID to series.
        let idToSeries = _chart.dataset._seriesMappingByHyperparameter[_chart.variantAttribute];
        // Get number of datapoints in series.
        // Caution: Assumes that all series have the same number of points.
        const numberOfRecordsInSeries = Object.keys(idToSeries.recordToSeriesMapping).length / idToSeries.seriesCount;
        // Collect best- and worst-performing parametrizations for all variations of variant parameter.
        let extrema = {};

        // Draw lines between points belonging to the same series.
        // Note: For quadratic interpolation see
        // https://stackoverflow.com/questions/7054272/how-to-draw-smooth-curve-through-n-points-using-javascript-html5-canvas.
        context.beginPath();
        for (let i = 0; i < idToSeries.seriesCount; i++) {
            // Retrieve datasets in this series.
            let records = [];
            for (let j = 0; j < idToSeries.seriesToRecordMapping[i].length; j++) {
                // Get data element by its ID.
                let record = _chart.dataset.getDataByID(
                    // Retrieve j-th element's ID from i-th series' ID mapping dictionary.
                    idToSeries.seriesToRecordMapping[i][j]
                );

                // Only add record if point is available (i. e. in active set of datapoints).
                if (record.id in dataPointCoordinates)
                    records.push(record);
            }

            // Sort records in series (list of dictionaries) )by attribute.
            // We don't know whether the attribute is numeric or not, so we don't use arithmetic operations to directly
            // determine the sort order. See
            // https://stackoverflow.com/questions/1129216/sort-array-of-objects-by-string-property-value-in-javascript.
            records.sort(function(a, b) {
                return (
                    a[_chart.variantAttribute] > b[_chart.variantAttribute]) ? 1 : (
                        (b[_chart.variantAttribute] > a[_chart.variantAttribute]) ? -1 : 0
                    );
            });

            // From first to pen-ultimate point in series: Draw line from one point to the next.
            for (let j = 0; j < records.length; j++) {
                // Get coordinates.
                let coordinatesStart    = dataPointCoordinates[records[j].id];

                // Only draw lines if loop hasn't arrived at last point.
                if (j < records.length - 1) {
                    let coordinatesEnd  = dataPointCoordinates[records[j + 1].id];

                    // Draw line between start and end point (if end point was available).
                    context.moveTo(coordinatesStart[0], coordinatesStart[1]);
                    context.lineTo(coordinatesEnd[0], coordinatesEnd[1]);
                }

                // Add information about possible extrema (for determining pareto frontiers)..
                let variantAttributeValue = records[j][variantAttribute];
                if (!(variantAttributeValue in extrema)) {
                    extrema[variantAttributeValue] = {min: coordinatesStart, max: coordinatesStart};
                }
                else {
                    if (coordinatesStart[1] < extrema[variantAttributeValue].min[1]) {
                        extrema[variantAttributeValue].min = coordinatesStart;
                    }
                    else if (coordinatesStart[1] > extrema[variantAttributeValue].max[1]) {
                        extrema[variantAttributeValue].max = coordinatesStart;
                    }
                }
            }
        }
        context.stroke();

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
            context.moveTo(extrema[key].min[0], extrema[key].min[1]);
            context.lineTo(extrema[nextKey].min[0], extrema[nextKey].min[1]);
            context.stroke();

            // Draw lines for pareto-pessimal/-optimal points.
            context.beginPath();
            context.moveTo(extrema[key].max[0], extrema[key].max[1]);
            context.lineTo(extrema[nextKey].max[0], extrema[nextKey].max[1]);
            context.stroke();
        }
    }

    /**
     * Plots all points in scatterplot.
     * @param context
     * @param data
     */
    function plotPointsOnCanvas(context, data, filteredDatapointCoordinates)
    {
        context.save();

        // ---------------------------------------------------------------------------------
        // 1. Gather data and separate in blocks with identical appearance/canvas settings
        // for performance optimization before plotting them.
        // ---------------------------------------------------------------------------------

        // then plot points - avoid unnecessary canvas state changes.
        let dataPoints = {
            filtered: {
                color: _chart.getColor(data[0]),
                radius: canvasElementSize(data[0], true),
                opacity: _nonemptyOpacity,
                coordinates: []
            },
            notFiltered: {
                color: _emptyColor,
                radius: canvasElementSize(data[0], false),
                opacity: _chart.excludedOpacity(),
                coordinates: []
            }
        };

        data.forEach(function (d, i) {
            for (let datapoint of d.value.items) {
                // Round coordinates to increase performance (avoid sub-pixel coordinates).
                if (datapoint.id in filteredDatapointCoordinates)
                    dataPoints.filtered.coordinates.push(filteredDatapointCoordinates[datapoint.id]);
                else {
                    dataPoints.notFiltered.coordinates.push([
                        _chart.x()(_chart.keyAccessor()(d)),
                        _chart.y()(_chart.valueAccessor()(d))
                    ]);
                }
            }
        });

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
        context.globalAlpha = 0; //dataPoints.notFiltered.opacity;
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
