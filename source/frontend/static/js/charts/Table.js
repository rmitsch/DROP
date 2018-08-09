import Chart from "./Chart.js";
import Utils from "../Utils.js";

export default class Table extends Chart
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
        // ----------------------------------------
        // 1. Initializing attributes.
        // ----------------------------------------

        super(name, panel, attributes, dataset, style, parentDivID);

        // Update involved CSS classes.
        $("#" + this._target).addClass("filter-reduce-table");

        // Initialize chart handler (too lazy to write a new class just for that).
        this._cf_chart = {};

        // Create div structure.
        let tableID = this._createDivStructure();

        // Select dimension to use for later look-ups.
        this._dimension = this._dataset.cf_dimensions[this._dataset.metadata.hyperparameters[0].name];

        // Create storage for filtered IDs.
        this._filteredIDs       = null;
        // Defines whether filter has already been added to jQuery's DataTable.
        this._filterHasBeenSet  = false;

        // ----------------------------------------
        // 2. Calling initialization procedures.
        // ----------------------------------------

        // Generate table.
        this._constructFCChart(tableID);

        // Implement methods necessary for dc.js hook and integrate it into it's chart registry.
        this._registerChartInDC();

        // Fill table initially.
        this._initTableData();
    }

    /**
     * Fetches initial set of data for table. Includes all datasets filtered in arbitrary dimension.
     * Assumption: Data is not filtered at initialization.
     * @private
     */
    _initTableData()
    {
        let records             = this._dimension.top(Infinity);
        let transformedRecords  = [records.length];

        // Transform records to format accepted by DataTable.
        for (let i = 0; i < records.length; i++) {
            let transformedRecord   = [this._attributes.length + 1];
            transformedRecord[0]    = records[i].id;
            for (let j = 0; j < this._attributes.length; j++) {
                transformedRecord[j + 1] = records[i][this._attributes[j]];
            }
            transformedRecords[i] = transformedRecord;
        }

        this._cf_chart.rows.add(transformedRecords);
        this._cf_chart.draw();
    }

    /**
     * Constructs DataTable object.
     * @param tableID
     * @private
     */
    _constructFCChart(tableID)
    {
        this._cf_chart = $("#" + tableID).DataTable({
            scrollX: true,
            scrollY: false,
            fixedColumns: false
        });

        // Highlight data point on hover in scatterplots & histograms.
        let instance = this;
        // $("#" + tableID + " tbody").on('mouseenter', 'td', Utils.debounce(function () {
        //     if (instance._cf_chart.row(this).data() !== null)
        //         instance._panel._operator.highlight(
        //                 instance._cf_chart.row(this).data()[0], instance._target
        //         );
        // }, 0));
        // $("#" + tableID + " tbody").on('mouseout', 'td', Utils.debounce(function () {
        //     // Clear highlighting.
        //     instance._panel._operator.highlight(
        //         null, instance._target
        //     );
        // }, 0));
        // On (double-)click: Open detail view.
        $("#" + tableID + " tbody").on('dblclick', 'td', function (e) {
            var data = instance._cf_chart.row(this).data();
            let modelID = data[0];
        });
    }

    _createDivStructure()
    {
        // Create table.
        let table       = document.createElement('table');
        table.id        = Utils.uuidv4();
        table.className = "display";
        $("#" + this._target).append(table);

        // Create table header.
        let tableHeader = "<thead><tr><th>ID</th>";
        // Append all hyperparameter to table.
        for (let i = 0; i < this._attributes.length; i++) {
            tableHeader += "<th>" + this._attributes[i] + "</th>";
        }
        tableHeader += "</tr></thead>";
        $("#" + table.id).append(tableHeader);

        return table.id;
    }

    /**
     * Implement methods necessary for dc.js hook and integrate it into it's chart registry.
     */
    _registerChartInDC()
    {
        // --------------------------------
        // 1. Implement necessary elements
        // of dc.js' interface for charts.
        // --------------------------------

        let instance = this;

        this._cf_chart.render       = function() {
            // Redraw chart.
            instance._cf_chart.draw();
        };

        this._cf_chart.redraw       = function() {
            // Update filtered IDs.
            let records = instance._dimension.top(Infinity);
            instance._filteredIDs = new Set();
            for (let i = 0; i < records.length; i++) {
                instance._filteredIDs.add(records[i].id)
            }

            // Filter table data using an ugly hack 'cause DataTable.js can't do obvious things.
            // Add filter only if it doesn't exist yet.
            if (!this._filterHasBeenSet)
                $.fn.dataTableExt.afnFiltering.push(
                    // oSettings holds information that can be used to differ between different tables -
                    // might be necessary once several tables use different filters.
                    function (oSettings, aData, iDataIndex) {
                        return instance._filteredIDs.has(+aData[0]);
                    }
                );

            // Redraw chart.
            instance._cf_chart.draw();
        };

        //
        this._cf_chart.filterAll    = function() {
            // Reset brush.
            instance._cf_chart.draw();
        };

        // --------------------------------
        // 2. Register parcoords plot in
        // dc.js' registry.
        // --------------------------------

        // Use operators ID as group ID (all panels in operator use the same dataset and therefore should be notified if
        // filter conditions change).
        dc.chartRegistry.register(this._cf_chart, this._panel._operator._target);
    }

    highlight(id, source)
    {
        if (source !== this._target) {
        }
    }
}