import Chart from "./Chart.js";
import Utils from "../Utils.js";

export default class ModelOverviewTable extends Chart
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
        this._tableID = this._createDivStructure();

        // Select dimension of ID to use for later look-ups.
        this._dimension = this._dataset.cf_dimensions["id"];
        // Create storage for filtered IDs.
        this._filteredIDs           = new Set();
        this._previousFilteredIDs   = new Set();
        this._selectedRows          = {};
        // Defines whether filter has already been added to jQuery's DataTable.
        this._filterHasBeenSet  = false;

        // ----------------------------------------
        // 2. Calling initialization procedures.
        // ----------------------------------------

        // Generate table.
        this._constructFCChart(this._tableID);

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
            scrollY: Math.floor($("#" + this._panel._target).height()) + "px",
            fixedColumns: false
        });

        let instance    = this;
        const table     = $("#" + tableID + " tbody");
        const stage     = instance._panel._operator._stage;
        stage.addKeyEventListener(this, ModelOverviewTable.processKeyEvent);

        // On hover: Highlight data point on hover in scatterplots & histograms.
        table.on('mouseenter', 'tr', function () {
            if (instance._cf_chart.row(this).data() !== null)
                instance._panel.highlight(
                    instance._cf_chart.row(this).data()[0],
                    instance._name,
                    true
                );
            }
        );
        // Clear highlighting on mouseout.
        table.on('mouseout', 'tr', function () {
            instance._panel.highlight(null, instance._name, true);
        });

        // On (double-)click: Open detail view.
        table.on('dblclick', 'td', function (e) {
            // Instruct model detail operator to load data for the selected model.
            stage._operators["ModelDetail"].loadData(
                // Fetch model ID from first field in selected table row.
                instance._cf_chart.row(this).data()[0]
            );
        });

        // On click: Filter.
        table.on('click', 'td', function (e) {
            const row           = instance._cf_chart.row(this);
            const selectedID    = row.data()[0];

            if (stage.ctrlDown) {
                if (instance._filteredIDs.has(selectedID)) {
                    instance._filteredIDs.delete(selectedID);
                    instance._selectedRows[selectedID].removeClass('selected');
                    delete instance._selectedRows[selectedID];
                }
                else {
                    instance._filteredIDs.add(selectedID);
                    instance._selectedRows[selectedID] = $(row.nodes());
                    instance._selectedRows[selectedID].addClass('selected');
                }
            }
        });
    }

    /**
     * Resets global filter status after reset() button was clicked.
     */
    resetFilterStatus()
    {
        this._dimension.filterFunction(x => true);
        for (let d of this._dimension.top(Infinity))
            this._filteredIDs.add(d.id);
        this.propagateFilterChange(this, "id");

        this._panel._operator.filter(this._filteredIDs);
        this._panel._operator.render();
        this._previousFilteredIDs = new Set(this._filteredIDs);

        this._filteredIDs.clear();
        this._selectedRows = {};
    }

    /**
     * Auxiliary function updating global filter status; includes re-rendering.
     * Note: Ideally all of this should be taken care of in this.propagateFilterChange().
     * @private
     */
    _updateGlobalFilterStatus()
    {
        this._dimension.filterFunction(x => this._filteredIDs.has(x));
        this.propagateFilterChange(this, "id");

        // Manually trigger redraw of elements in this operator.
        this._panel._operator.filter(this._filteredIDs);
        this._panel._operator.render();
        this._previousFilteredIDs = new Set(this._filteredIDs);
    }

    /**
     * Processes key event. Called by stage object.
     * @param instance
     * @param keyEvent
     */
    static processKeyEvent(instance, keyEvent)
    {
        if (keyEvent.key === "Control" && keyEvent.ctrlKey === false) {
            // Reset row highlighting.
            for (let id in instance._selectedRows)
                instance._selectedRows[id].removeClass('selected');

            if (!Utils.eqSet(instance._filteredIDs, instance._previousFilteredIDs)) {
                if (instance._filteredIDs.size > 0) {
                    instance._updateGlobalFilterStatus();
                }
            }

            instance._filteredIDs.clear();
            instance._selectedRows = {};
        }
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

    render()
    {
        this._cf_chart.redraw();
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
                        // Check oSettings to see if we have to apply this filter. Otherwise ignore (i. e. return true
                        // for all elements).
                        return oSettings.sTableId === instance._tableID ?
                            instance._filteredIDs.has(+aData[0]) :
                            true;
                    }
                );

            // Redraw chart.
            instance._cf_chart.draw();
        };

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

    get table()
    {
        return this._cf_chart;
    }
}