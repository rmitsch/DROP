 // IDs of menu buttons.
menuIDs = ["menu_prototype", "menu_about"]

// Initialize setup UI.
$(document).ready(function() {
    // Fetch model metadata.
    $.ajax({
        url: '/get_metadata',
        type: 'GET',
        success: function(metadata_json) {
            // Parse delivered JSON with metadata for all models.
            metadata_json = JSON.parse(metadata_json);

            // Get information on which hyperparameters and objectives are available.
            $.ajax({
                url: '/get_metadata_template',
                type: 'GET',
                success: function(metadata_template_json) {
                    console.log(metadata_template_json["hyperparameters"]);
                    console.log(metadata_template_json["objectives"]);

                    // Next: Spawn instances of panels and components in UI.
                    // Start with hyperparameter/objective panel.
                    // All components inside a panel are automatically linked with dc.js. Panels have to be linked
                    // with each other explicitly, if so desired (since used datasets may differ).
                }
            });

            // TEMPORARY: Transform/fake existing records so that each record has a dictionary 'objectives' holding all
            // objectives, while other entries are hyperparameters.
            // Remove once change above is implemented and datasets based on that are used (0.2).
//            for (let i = 0; i < Object.keys(metadata_json).length; i++) {
//                metadata_json[i]["hyperparameters"] = {
//                    "n_components": metadata_json[i].n_components,
//                    "perplexity": metadata_json[i].perplexity,
//                    "early_exaggeration": metadata_json[i].early_exaggeration,
//                    "learning_rate": metadata_json[i].learning_rate,
//                    "n_iter": metadata_json[i].n_iter,
//                    // "min_grad_norm": metadata_json[i].min_grad_norm
//                    "angle": metadata_json[i].angle,
//                    "metric": metadata_json[i].metric
//                };
//                metadata_json[i]["objectives"] = {
//                    "runtime": metadata_json[i].runtime,
//                    "trustworthiness": metadata_json[i].trustworthiness,
//                    "continuity": metadata_json[i].continuity
//                };
//
//                delete metadata_json[i].n_components;
//                delete metadata_json[i].perplexity;
//                delete metadata_json[i].early_exaggeration;
//                delete metadata_json[i].learning_rate;
//                delete metadata_json[i].n_iter;
//                delete metadata_json[i].angle;
//                delete metadata_json[i].metric;
//                delete metadata_json[i].runtime;
//                delete metadata_json[i].trustworthiness;
//                delete metadata_json[i].continuity;
//            }


            // Next: Spawn instances of
        }
     });
});