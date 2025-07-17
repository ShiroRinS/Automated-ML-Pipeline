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
        return;
    }

    // Parse suggestions if it's a string
    let suggestionData;
    try {
        suggestionData = typeof suggestions === 'string' ? JSON.parse(suggestions) : suggestions;
    } catch (e) {
        $('#geminiSuggestions').html('<div class="alert alert-danger">Error parsing suggestions: ' + e.message + '</div>').show();
        return;
    }

    // Build the HTML content
    const html = `
        <div class="gemini-suggestions">
            ${Object.entries(suggestionData.response).map(([section, items]) => `
                <div class="suggestion-section">
                    <h3>${section}</h3>
                    <ul class="suggestion-list">
                        ${items.map(item => `
                            <li class="suggestion-item">${item}</li>
                        `).join('')}
                    </ul>
                </div>
            `).join('')}
            
            <div class="metadata-footer">
                <div class="metadata-item">
                    <i class="fas fa-tag"></i> ${suggestionData.metadata.topic}
                </div>
                <div class="metadata-item">
                    <i class="fas fa-clock"></i> ${new Date(suggestionData.metadata.timestamp).toLocaleString()}
                </div>
                <div class="metadata-item">
                    <i class="fas fa-info-circle"></i> ${suggestionData.metadata.context}
                </div>
            </div>
        </div>
    `;

    $('#geminiSuggestions').html(html).show();
}

// Update selected features list
function updateSelectedFeatures() {
    var selectedFeatures = $('.feature-item.selected').map(function() {
        return $(this).find('h6').text();
    }).get();
    
    // Update the count
    $('#selectedCount').text(selectedFeatures.length + ' selected');
    
    // Update the selected features display
    var html = '';
    selectedFeatures.forEach(function(feature) {
        html += `
            <div class="badge bg-primary p-2 me-2 mb-2">
                ${feature}
                <button type="button" class="btn-close btn-close-white ms-2" 
                    onclick="deselectFeature('${feature}')"></button>
            </div>
        `;
    });
    $('#selectedFeatures').html(html);
    
    // Show/hide the no features message
    if (selectedFeatures.length === 0) {
        $('#noFeaturesSelected').show();
    } else {
        $('#noFeaturesSelected').hide();
    }
}

// Deselect a feature
function deselectFeature(featureName) {
    $(`.feature-item:contains('${featureName}')`).removeClass('selected');
    updateSelectedFeatures();
}

// Initialize feature selection functionality
function initializeFeatureSelection() {
    console.log('Initializing feature selection...');
    
    // Show numeric features only
    $('#showNumeric').click(function() {
        $('.feature-item').hide();
        $('.feature-item[data-type="numeric"]').show();
        $('.btn-group .btn').removeClass('active');
        $(this).addClass('active');
    });
    
    // Show categorical features only
    $('#showCategorical').click(function() {
        $('.feature-item').hide();
        $('.feature-item[data-type="categorical"]').show();
        $('.btn-group .btn').removeClass('active');
        $(this).addClass('active');
    });
    
    // Show all features
    $('#showAll').click(function() {
        $('.feature-item').show();
        $('.btn-group .btn').removeClass('active');
        $(this).addClass('active');
    });
    
    // Initialize feature click handlers
    $('.feature-item').on('click', function() {
        console.log('Feature clicked:', $(this).find('h6').text());
        $(this).toggleClass('selected');
        updateSelectedFeatures();
    });
    
    // Initially show all features and highlight the 'All' button
    $('#showAll').addClass('active');
    $('.feature-item').show();
}

// Initialize page
$(document).ready(function() {
    console.log('Document ready');
    
    // Initialize recommendations if not already defined
    if (typeof initial_recommendations === 'undefined') {
        window.initial_recommendations = null;
    }
    
// Check for data analysis
if (typeof data_analysis !== 'undefined' && data_analysis) {
    console.log('Data analysis:', data_analysis);
    showDataSummary(data_analysis);
    showAvailableFeatures(data_analysis);
    
    // Initialize feature selection handlers
    $('.feature-item').on('click', function() {
        $(this).toggleClass('selected');
        updateSelectedFeatures();
    });
}
        console.log('No data analysis available');
    }
    
    // Check if initial recommendations are available
    if (window.initial_recommendations) {
        console.log('Initial recommendations:', window.initial_recommendations);
        // Show Gemini suggestions if available
        if (window.initial_recommendations.raw_suggestions) {
            showGeminiSuggestions(window.initial_recommendations.raw_suggestions);
        }
        
        // Show feature importance if available
        if (window.initial_recommendations.feature_importances) {
            createFeatureImportanceChart(window.initial_recommendations.feature_importances);
        }
    }
});
