import PrototypeStage from './stages/PrototypeStage.js';
import Utils from './Utils.js'
import DRMetaDataset from "./data/DRMetaDataset.js";

// IDs of menu buttons.
let menuIDs = ["menu_prototype", "menu_about"];

// Initialize setup UI.
$(document).ready(function() {
    let now = new Date();
    console.log("*** DROP *** Starting construction at " + now.getHours() + ":" + now.getMinutes() + ":" + now.getSeconds() + ".");

    // -----------------------------------------------------
    // 1. Process GET parameters.
    // -----------------------------------------------------

    let metadataGETParameters = processGETParameters();

    // -----------------------------------------------------
    // 2. Initialize dataset loading button.
    // -----------------------------------------------------

    $("#load-dataset-link").click(function() {
        let baseURL = location.protocol + '//' + location.hostname + (location.port ? ':'+location.port: '');
        // Get content of dropdowns, preprocess for backend.
        let datasetNameTranslation = {
            Wine: "wine",
            MNIST: "mnist",
            "Swiss Roll": "swiss_roll",
            "VIS Papers": "vis",
            Happiness: "happiness"
        };
        let drkTranslation = {"t-SNE": "tsne", UMAP: "umap", SVD: "svd"};
        // Load new page.
        window.location.href =
            baseURL + "?dataset=" +
            datasetNameTranslation[$("#datasetLink").html()] + "&drk=" +
            drkTranslation[$("#drkernelLink").html()];
    });

    // -----------------------------------------------------
    // 3. Fetch model metadata - both structure and content.
    // -----------------------------------------------------

    console.log("Fetch metadata.");
    $.ajax({
        url: '/get_metadata',
        data: metadataGETParameters,
        type: 'GET',
        success: function(model_data) {
            // Parse delivered JSON with metadata for all models.
            model_data = JSON.parse(model_data);

            // Cast Object to array.
            let model_data_list = [];
            for (let key in model_data) {
                if (model_data.hasOwnProperty(key)) {
                    // Add ID to entry, then add to list.
                    let currParametrization = model_data[key];
                    currParametrization["id"] = parseInt(key);
                    model_data_list.push(currParametrization);
                }
            }

            // Get information on which hyperparameters and objectives are available.
            // Note: Sequence of calls is important, since /get_metadata_template uses information
            // made available by /get_metadata.
            $.ajax({
                url: '/get_metadata_template',
                type: 'GET',
                // Compile or load DRMetadatset.
                success: function(model_metadata) {
                    let datasetStorageID    = "test"
                    let dataset             = localStorage.getItem("someVarKey");

                    if (false && typeof dataset !== "undefined") {
                        console.log("Loaded DRMetaDataset.");
                        console.log(dataset)
                    }
                    else {
                        console.log("Compiling and storing DRMetaDataset.");
                        dataset = new DRMetaDataset(
                            "PrototypeDataset",
                            model_data_list,
                            model_metadata,
                            10
                        );

                        // Store object in JSON.
                        // Session.set(datasetStorageID, dataset);
                        // localStorage.setItem("someVarKey", dataset);
                    }


                    // All components inside a panel are automatically linked with dc.js. Panels have to be linked
                    // with each other explicitly, if so desired (since used datasets may differ).
                    console.log("Constructing stage.");
                    let prototypeStage = new PrototypeStage(
                        "PrototypeStage",
                        "stage",
                        {
                            modelMetadata: dataset,
                            surrogateModel: null,
                            dissonance: null
                        }
                    );
                }
            });
        }
     });
});

/**
 * Reads GET parameters defining which dataset and kernel to use.
 * Defaults to wine dataset + t-SNE, if none specified.
 * @returns {{datasetName: *, drKernelName: *}}
 */
function processGETParameters()
{
    // Read GET parameters.
    let datasetName     = Utils.findGETParameter("dataset") === null ? "happiness" : Utils.findGETParameter("dataset");
    let drKernelName    = Utils.findGETParameter("drk") === null ? "tsne" : Utils.findGETParameter("drk");
    let forceReload     = Utils.findGETParameter("force") === null ? "false" : Utils.findGETParameter("force");

    // Update displayed value of dropdown based on current URL parameters.
    let datasetNameTranslation = {
        wine: "Wine", mnist: "MNIST", swiss_roll: "Swiss Roll", vis: "VIS Papers", happiness: "Happiness"
    };
    let drkTranslation = {tsne: "t-SNE", umap: "UMAP", svd: "SVD"};
    $("#datasetLink").html(datasetNameTranslation[datasetName]);
    $("#drkernelLink").html(drkTranslation[drKernelName]);

    return {
        datasetName: datasetName,
        drKernelName: drKernelName,
        forceReload: forceReload === "true"
    }
}