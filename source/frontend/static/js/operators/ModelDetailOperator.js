import Operator from "./Operator.js";
import ModelDetailPanel from "../panels/ModelDetailPanel.js";
import ModelDetailDataset from "../data/ModelDetailDataset.js";

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
     * @param drMetaDataset Instance of DRMetaDataset used for FilterReduce operator.
     * @param parentDivID
     */
    constructor(name, stage, drMetaDataset, parentDivID)
    {
        super(name, stage, "1", "1", null, parentDivID);
        this._modelID       = null;
        this._drMetaDataset = drMetaDataset;

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
        this._modelID   = modelID;
        let scope       = this;

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
            scope._dataset = new ModelDetailDataset(
                "Model Detail Data", modelID, modelDetailData, scope._drMetaDataset
            );

            // Prompt panel to update data and to re-render.
            scope._panels["Model Details"].update();
        });
    }

    filter(embeddingIDs)
    {
        // Note: Filtering on global level has no impact on MDO.
    }
}