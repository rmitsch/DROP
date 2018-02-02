import PrototypeView from './views/PrototypeView.js';

// IDs of menu buttons.
let menuIDs = ["menu_prototype", "menu_about"]

// Initialize setup UI.
$(document).ready(function() {
    // Fetch model metadata.
    $.ajax({
        url: '/get_metadata',
        type: 'GET',
        success: function(model_data) {
            // Parse delivered JSON with metadata for all models.
            model_data = JSON.parse(model_data);

            // Get information on which hyperparameters and objectives are available.
            $.ajax({
                url: '/get_metadata_template',
                type: 'GET',
                success: function(model_metadata) {
                    console.log(model_metadata["hyperparameters"]);
                    console.log(model_metadata["objectives"]);

                    // Next: Spawn instances of panels and components in UI.
                    // Start with hyperparameter/objective panel.
                    // All components inside a panel are automatically linked with dc.js. Panels have to be linked
                    // with each other explicitly, if so desired (since used datasets may differ).
                    let prototypeView = new PrototypeView("PrototypeView", model_data, model_metadata);
                }
            });
        }
     });
});