import Operator from "./Operator.js";
import ModelDetailPanel from "../panels/ModelDetailPanel.js";

/**
 * Class for ModelDetailOperator.
 * One operator operates on exactly one dataset (-> one instance of a DR model, including detailled information - like
 * coordinates - on all its records).
 */
export default class ModelDetailOperator extends Operator
{
    /**
     * Constructs new ModelDetailOperator.
     * Note that at initialization time no dataset is required.
     * @param name
     * @param stage
     * @param parentDivID
     */
    constructor(name, stage, parentDivID)
    {
        super(name, stage, "1", "1", null, parentDivID);

        // Update involved CSS classes.
        $("#" + this._target).addClass("model-detail-operator");

        // Construct all necessary panels.
        this.constructPanels();
    }

    /**
     * Constructs all panels required by this operator.
     */
    constructPanels()
    {
        // ----------------------------------------------
        // Generate panels.
        // ----------------------------------------------

        // Construct panels for charts.
        let mdPanel = new ModelDetailPanel(
            "Model Details",
            this
        );
        this._panels[mdPanel.name] = mdPanel;
    }

    /**
     * Loads data and constructs dataset from specified DR model ID.
     * @param modelID
     */
    loadData(modelID)
    {
        let scope = this;

        // Fetch model data.
        fetch(
            "/get_dr_model_details?id=" + modelID,
            {
                headers: { "Content-Type": "application/json; charset=utf-8"},
                method: "GET"
            }
        )
        .then(res => res.json())
        .then(modelDetailData => {
            // Parse substructures.
            modelDetailData.model_metadata = JSON.parse(modelDetailData.model_metadata);
            modelDetailData.original_dataset = JSON.parse(modelDetailData.original_dataset);
            // Store dataset.
            scope._dataset = {
                primitiveData: modelDetailData,
                crossfilter: null
            };

            // Show modal.
            $("#" + scope._panels["Model Details"]._target).dialog({
                title: "Model Details for Model with ID #" + modelID,
                width: $("#" + scope._stage._target).width() / 1.5,
                height: $("#" + scope._stage._target).height() / 1.5
            });
        });

        // todo Pass data on to model detail panel, then to chart(s). (Re-)render.
    }
}