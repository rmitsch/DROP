import PrototypeStage from './stages/PrototypeStage.js';
import Dataset from './data/DRMetaDataset.js';
import Utils from './Utils.js'

// IDs of menu buttons.
let menuIDs = ["menu_prototype", "menu_about"]

// Initialize setup UI.
$(document).ready(function() {
    // Fetch model metadata.
    $.ajax({
        url: '/get_metadata',
        data: {
            // Read GET parameters.
            "datasetName": Utils.findGETParameter("dataset"),
            "drKernelName": Utils.findGETParameter("drk")
        },
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
                success: function(model_metadata) {
                    // Generate dataset.
                    let dataset = new Dataset("PrototypeDataset", model_data_list, model_metadata, 5);

                    // All components inside a panel are automatically linked with dc.js. Panels have to be linked
                    // with each other explicitly, if so desired (since used datasets may differ).
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