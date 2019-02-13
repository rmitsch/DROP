import Chart from "./Chart.js";
import Utils from "../Utils.js"


/**
 * Creates chart for surrogate model.
 * Supported so far: Decision tree.
 * Code for tree: https://bl.ocks.org/ajschumacher/65eda1df2b0dd2cf616f.
 * Alternative: http://bl.ocks.org/pprett/3813537.
 */
export default class SurrogateModelChart extends Chart
{
    /**
     *
     * @param name
     * @param panel
     * @param attributes Ignored.
     * @param dataset
     * @param style
     * @param parentDivID
     */
    constructor(name, panel, attributes, dataset, style, parentDivID)
    {
        super(name, panel, attributes, dataset, style, parentDivID);

        // Construct graph.
        this.constructCFChart();
    }

    /**
     * Construction happens at rendering time.
     */
    constructCFChart()
    {
    }

    /**
     * Resets canvas and updates dataset.
     * @param dataset
     */
    reset(dataset)
    {
        this._dataset = dataset;
        // Remove SVG.
        d3.select("#surrogate-model-chart-svg").remove();
    }

    /**
     * Constructs and draws chart drawing decision tree.
     * Source: https://bl.ocks.org/ajschumacher/65eda1df2b0dd2cf616f.
     */
    render()
    {
        try {
            let operatorDiv = $("#" + this._panel._operator._target);
            this._lastPanelSize.width = operatorDiv.width();
            this._lastPanelSize.height = operatorDiv.height();

            let margin      = {top: 5, right: 120, bottom: 20, left: 120};
            let baseWidth   = operatorDiv.width() - 0;
            let baseHeight  = operatorDiv.height() - 5;
            let treeData    = JSON.parse(JSON.stringify(this._dataset));
            let scope       = this;

            // Generate chart.
            let width   = baseWidth - margin.right - margin.left;
            let height  = baseHeight - margin.top - margin.bottom;

            let i = 0,
                duration = 500,
                root;

            let tree = d3.layout.tree()
                .size([height, width]);

            let diagonal = d3.svg.diagonal()
                .projection(function (d) {
                    return [d.y, d.x];
                });

            let svgContainer = d3.select("#" + this._target);
            // Create chart.
            let svg = svgContainer.append("svg")
                .attr("id", "surrogate-model-chart-svg")
                .attr("width", "100%")
                .attr("height", height + margin.top + margin.bottom)
                .append("g")
                .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

            root = treeData;
            root.x0 = height / 2;
            root.y0 = 0;

            function collapse(d) {
                if (d.children) {
                    d._children = d.children;
                    d._children.forEach(collapse);
                    d.children = null;
                }
            }

            root.children.forEach(collapse);
            update(root);

            // d3.select(self.frameElement).style("height", "480px");

            function update(source) {
                // Compute the new tree layout.
                let nodes = tree.nodes(root).reverse(),
                    links = tree.links(nodes);

                // Normalize for fixed-depth.
                nodes.forEach(function (d) {
                    d.y = d.depth * 110;
                });

                // Update the nodes…
                let node = svg.selectAll("g.node")
                    .data(nodes, function (d) {
                        return d.id || (d.id = ++i);
                    });

                // Enter any new nodes at the parent's previous position.
                let nodeEnter = node.enter().append("g")
                    .attr("class", "node")
                    .attr("transform", function (d) {
                        return "translate(" + source.y0 + "," + source.x0 + ")";
                    })
                    .on("click", click);

                nodeEnter.append("circle")
                    .attr("r", 1e-6)
                    .style("fill", function (d) {
                        return d._children ? "lightsteelblue" : "#fff";
                    });

                nodeEnter.append("text")
                    .attr("x", function (d) {
                        return d.children || d._children ? -10 : 10;
                    })
                    .attr("dy", ".35em")
                    .attr("text-anchor", function (d) {
                        return d.children || d._children ? "end" : "start";
                    })
                    .text(function (d) {
                        return d.name;
                    })
                    .style("fill-opacity", 1e-6);

                // Transition nodes to their new position.
                let nodeUpdate = node.transition()
                    .duration(duration)
                    .attr("transform", function (d) {
                        return "translate(" + d.y + "," + d.x + ")";
                    });

                nodeUpdate.select("circle")
                    .attr("r", 4.5)
                    .style("fill", function (d) {
                        return d._children ? "lightsteelblue" : "#fff";
                    });

                nodeUpdate.select("text")
                    .style("fill-opacity", 1);

                // Transition exiting nodes to the parent's new position.
                let nodeExit = node.exit().transition()
                    .duration(duration)
                    .attr("transform", function (d) {
                        return "translate(" + source.y + "," + source.x + ")";
                    })
                    .remove();

                nodeExit.select("circle")
                    .attr("r", 1e-6);

                nodeExit.select("text")
                    .style("fill-opacity", 1e-6);

                // Update the links…
                let link = svg.selectAll("path.link")
                    .data(links, function (d) {
                        return d.target.id;
                    });

                // Enter any new links at the parent's previous position.
                link.enter().insert("path", "g")
                    .attr("class", "link")
                    .attr("d", function (d) {
                        let o = {x: source.x0, y: source.y0};
                        return diagonal({source: o, target: o});
                    });

                // Transition links to their new position.
                link.transition()
                    .duration(duration)
                    .attr("d", diagonal);

                // Transition exiting nodes to the parent's new position.
                link.exit().transition()
                    .duration(duration)
                    .attr("d", function (d) {
                        let o = {x: source.x, y: source.y};
                        return diagonal({source: o, target: o});
                    })
                    .remove();

                // Stash the old positions for transition.
                nodes.forEach(function (d) {
                    d.x0 = d.x;
                    d.y0 = d.y;
                });
            }

            // Toggle children on click.
            function click(d) {
                if (d.children) {
                    d._children = d.children;
                    d.children = null;
                } else {
                    d.children = d._children;
                    d._children = null;
                }
                update(d);

                // Widen panel, if necessary.
                if (d.children) {
                    let panelDiv = $("#" + scope._panel._target);
                    // Add margin.left and 120 to reflect offset on the left and length of result label on the right.
                    let value = (d.depth + 1) * 110 + margin.left + 120;
                    let currPanelWidth = panelDiv.width();

                    if (currPanelWidth < value) {
                        panelDiv.width(value);
                    }
                }

                // Shrink panel, if necessary.
                else if (d._children) {
                    //scope._extrema.x[1] = d.x + margin.left;

                    // todo: Shrink panel, if necessary. Requires checking x-values of other nodes.
                    // let currPanelWidth = $("#" + scope._panel._target).width();
                    // if (currPanelWidth > maxX) {
                    //     console.log("shrinking from " + currPanelWidth + " to " + maxX);
                    //     $("#" + scope._panel._target).width(maxX);
                    // }
                }
            }
        }

        catch (exception) {
            console.log("SurrogateModelChart couldn't render.")
        }
    }

    resize()
    {
        let operatorDiv = $("#" + this._panel._operator._target);

        // Check if panel height has changed.
        if (operatorDiv.height() != this._lastPanelSize.height) {
            d3.select("#surrogate-model-chart-svg").remove();
            this.render();
        }
    }

     /**
      * Create (hardcoded) div structure for child nodes.
      * @deprecated
      * @returns {Object}
     */
    _createDivStructure()
    {
        // -----------------------------------
        // Create charts container.
        // -----------------------------------

        let treeDiv = Utils.spawnChildDiv(this._target, null, "surrogate-model-chart");

        return {
            treeDivID: treeDiv.id
        };
    }
}