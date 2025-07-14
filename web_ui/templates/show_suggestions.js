function showGeminiSuggestions(suggestions) {
    $('#geminiSpinner').hide();
    if (typeof suggestions === 'string' && suggestions.startsWith('Error')) {
        $('#geminiSuggestions').html('<div class="alert alert-danger">' + suggestions + '</div>').show();
    } else {
        // Configure marked options
        marked.setOptions({
            breaks: true,  // Enable line breaks
            gfm: true,    // Enable GitHub Flavored Markdown
            headerIds: false,
            mangle: false
        });
        
        // Render markdown content
        const htmlContent = marked.parse(suggestions);
        
        $('#geminiSuggestions').html(`
            <div class="alert alert-info markdown-content">
                ${htmlContent}
            </div>
        `).show();
    }
}
