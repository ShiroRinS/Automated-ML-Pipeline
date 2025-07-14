// Feature importance visualization functions
function createFeatureImportanceChart(importances) {
    if (!importances || importances.length === 0) {
        $('#featureImportanceChart').html('<div class="alert alert-warning">No feature importance data available</div>');
        return;
    }

    // Sort features by importance
    importances.sort((a, b) => b.importance - a.importance);

    // Add explanation at the top
    var html = `
        <div class="alert alert-info mb-3">
            <h6 class="mb-2"><i class="fas fa-info-circle me-2"></i>Feature Importance Guide</h6>
            <p class="mb-0" style="font-size: 0.9em;">
                Percentages show each feature's relative importance in predicting survival, calculated using Random Forest (100 trees, random_state=42) with Gini impurity reduction.
                Higher percentages indicate stronger predictive power. These scores are based on how much each feature helps in making accurate predictions.
                <br><small class="text-muted mt-1 d-block">Model: RandomForestClassifier(n_estimators=100, random_state=42)</small>
            </p>
        </div>
        <div class="list-group">`;

    // Create feature list with importance scores
    importances.forEach(function(item) {
        var importance = (item.importance * 100).toFixed(2);
        html += `
            <div class="list-group-item">
                <div class="d-flex w-100 justify-content-between align-items-center">
                    <div>
                        <h6 class="mb-1">${item.feature}</h6>
                        <small class="text-muted">Predictive Power: ${importance}%</small>
                    </div>
                    <span class="badge bg-primary">#${importances.indexOf(item) + 1}</span>
                </div>
                <div class="progress mt-2" style="height: 8px;">
                    <div class="progress-bar bg-primary" role="progressbar" 
                         style="width: ${importance}%" 
                         title="${importance}% importance">
                    </div>
                </div>
            </div>`;
    });

    html += '</div>';
    $('#featureImportanceChart').html(html);
}

// Gemini suggestions display function
function showGeminiSuggestions(suggestions) {
    $('#geminiSpinner').hide();
    if (typeof suggestions === 'string' && suggestions.startsWith('Error')) {
        $('#geminiSuggestions').html('<div class="alert alert-danger">' + suggestions + '</div>').show();
    } else {
        // Parse markdown to HTML using marked library
        const htmlContent = marked.parse(suggestions);
        
        $('#geminiSuggestions').html(`
            <div class="alert alert-info markdown-content">
                ${htmlContent}
            </div>
        `).show();
    }
}

// Initialize page
$(document).ready(function() {
    // Check if initial recommendations are available
    if (typeof initial_recommendations !== 'undefined' && initial_recommendations) {
        // Show Gemini suggestions
        showGeminiSuggestions(initial_recommendations.raw_suggestions);
        
        // Show feature importance
        createFeatureImportanceChart(initial_recommendations.feature_importances);
    }
});
