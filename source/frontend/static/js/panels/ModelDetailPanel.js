import Panel from "./Panel.js";
import Utils from "../Utils.js";

/**
 * Panel for model detail view.
 */
export default class ModelDetailPanel extends Panel
{
    /**
     * Constructs new panel for model detail view charts.
     * @param name
     * @param operator
     * @param parentDivID
     */
    constructor(name, operator, parentDivID)
    {
        super(name, operator, parentDivID);

        // Update involved CSS classes.
        $("#" + this._target).addClass("model-detail-panel");

        // Create div structure for child nodes.
        this._divStructure = this._createDivStructure();

        // Generate charts.
        this._generateCharts();
    }

    /**
     * Generates all chart objects. Does _not_ render them.
     */
    _generateCharts()
    {
        console.log("Generating ModelDetailPanel...");

        console.log(this._operator)
        // var values = [500, 400, 700, 900, 1200, 300, 550];
        // var dates = {};
        //
        // var counter = 0;
        // for (var i = values.length - 1; i >= 0; i--) {
        //     var d = Math.random();
        //     dates[counter] = d;
        //     counter++;
        // }
        // console.log(dates)
        //
        // $("#sparklinetest").sparkline(values, {
        //     type: "bar",
        //     barWidth: 20,
        //     barSpacing: 3,
        //     height: 100,
        //     tooltipValueLookups: {
        //         names: {
        //             0: 'Squirrel',
        //             1: 'Kitty',
        //             2: 'Bird',
        //             3: 'Three',
        //             4: 'Four',
        //             5: 'Five',
        //             6: 'Six',
        //             7: 'Seven'
        //             // Add more here
        //         }},
        //     colorMap: ["green", "blue", "blue", "blue", "blue", "blue", "red"]
        // });
    }

    /**
     * Create (hardcoded) div structure for child nodes.
     * @returns {Object}
     */
    _createDivStructure()
    {
        // -----------------------------------
        // 1. Create panes.
        // -----------------------------------

        // Left pane.
        let parameterPane = Utils.spawnChildDiv(this._target, "model-detail-parameter-pane", "split split-horizontal");
        // Right pane.
        let samplePane = Utils.spawnChildDiv(this._target, "model-detail-sample-pane", "split split-horizontal");

        // Upper-left pane - hyperparameters and objectives for current DR model.
        let attributePane = Utils.spawnChildDiv(
            parameterPane.id, null, "model-detail-pane split split-vertical",
            `<div class='model-details-block reduced-padding'>
                <div class='model-details-title'>Hyperparameters</div>
                <hr>
                <div class='model-details-title'>Objectives</span>
            </div>`
        );
        // Bottom-left pane - explanation of hyperparameter importance for this DR model utilizing LIME.
        let limePane = Utils.spawnChildDiv(
            parameterPane.id, null, "model-detail-pane split-vertical",
            `<div class='model-details-block'>
                <div class='model-details-title'>Local Hyperparameter Relevance</div>
            </div>`
        );

        // Upper-right pane - all records in scatterplot (SPLOM? -> What to do with higher-dim. projections?).
        let scatterplotPane = Utils.spawnChildDiv(
            samplePane.id, null, "model-detail-pane split-vertical",
            `<div class='model-details-block reduced-padding'>
                <div class='model-details-title'>All Records</div>
            </div>`
        );
        // Bottom-right pane - detailed information to currently selected record.
        let recordPane = Utils.spawnChildDiv(
            samplePane.id, null, "model-detail-pane split-vertical",
            `<div class='model-details-block'>
                <div class='model-details-title'>Selected Sample(s)</span>
            </div>`
        );

        // -----------------------------------
        // 2. Configure splitting.
        // -----------------------------------

        // Split left and right pane.
        Split(["#" + parameterPane.id, "#" + samplePane.id], {
            direction: "horizontal",
            sizes: [25, 75],
            onDragEnd: function() {}
        });

        // Split upper-left and bottom-left pane.
        Split(["#" + attributePane.id, "#" + limePane.id], {
            direction: "vertical",
            sizes: [50, 50],
            onDragEnd: function() {}
        });

        // Split upper-right and bottom-right pane.
        Split(["#" + scatterplotPane.id, "#" + recordPane.id], {
            direction: "vertical",
            sizes: [50, 50],
            onDragEnd: function() {}
        });

        // Return all panes' IDs.
        return {
            parameterPane: parameterPane.id,
            samplePane: samplePane.id,
            attributePane: attributePane.id,
            limePane: limePane.id,
            scatterplotPane: scatterplotPane.id,
            recordPane: recordPane.id
        };
    }

    render()
    {
        // this._chart.render();
    }

    processSettingsChange(delta)
    {
        // this._chart.orderBy(delta);
    }
}