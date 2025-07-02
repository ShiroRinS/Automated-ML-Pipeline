# 🌐 ML Pipeline Web UI

A modern, responsive web interface for viewing and managing ML pipeline results, built with Flask and Bootstrap.

## 🚀 Quick Start

1. **Install dependencies**:
```bash
pip install flask jinja2
```

2. **Start the web server**:
```bash
python web_ui/app.py
```

3. **Open your browser**:
```
http://localhost:5000
```

## 📊 Features

### **🏠 Dashboard**
- **System Overview**: Total models, predictions, training sessions
- **Performance Metrics**: Best model score and recent activity
- **Quick Stats**: Real-time system health indicators

### **🔮 Predictions**
- **File Browser**: List all prediction results with metadata
- **Detailed Views**: Rich prediction analysis with summaries
- **Download Support**: Direct CSV file downloads
- **Statistics**: Survival rates, class distributions

### **📚 Training History**
- **Session Tracking**: Complete history of all training runs
- **Performance Comparison**: Test scores, CV results, feature counts
- **Model Status**: Active/inactive model indicators
- **Detailed Logs**: Full training session information

### **🔧 Models**
- **Model Management**: Overview of all trained models
- **Metadata Display**: Feature counts, target variables, file sizes
- **Version Tracking**: Unique training IDs and timestamps

### **❤️ System Status**
- **Health Checks**: Directory and file integrity monitoring
- **Recent Activity**: Timeline of training and prediction events
- **Resource Usage**: File sizes and system statistics

## 🎨 UI Components

### **Navigation**
- **Sidebar Navigation**: Easy access to all sections
- **Active Indicators**: Current page highlighting
- **Responsive Design**: Works on desktop and mobile

### **Data Visualization**
- **Metric Cards**: Color-coded system statistics
- **Interactive Tables**: Sortable and responsive data tables
- **Summary Panels**: Quick insights with key metrics
- **Status Badges**: Visual indicators for model status

### **User Experience**
- **Real-time Updates**: Auto-refreshing timestamps
- **Error Handling**: Graceful error pages with navigation
- **Smooth Animations**: Hover effects and transitions
- **Professional Styling**: Modern gradient design

## 📁 Project Structure

```
web_ui/
├── app.py                 # Flask application main file
├── templates/             # Jinja2 HTML templates
│   ├── base.html         # Base template with navigation
│   ├── index.html        # Dashboard page
│   ├── predictions.html   # Predictions listing
│   ├── prediction_detail.html  # Individual prediction view
│   ├── training_history.html   # Training history table
│   └── error.html        # Error page
└── README.md             # This documentation
```

## 🔧 API Endpoints

### **Web Pages**
- `GET /` - Main dashboard
- `GET /predictions` - Prediction results listing
- `GET /predictions/<filename>` - Detailed prediction view
- `GET /training-history` - Training session history
- `GET /training-detail/<training_id>` - Detailed training info
- `GET /models` - Available models overview
- `GET /system-status` - System health status

### **API & Downloads**
- `GET /api/stats` - JSON system statistics
- `GET /download/prediction/<filename>` - Download prediction CSV

## 🎯 Key Features

### **Real-time Data**
- Automatically loads latest predictions and training results
- Live system statistics and health monitoring
- Dynamic content updates without page refresh

### **Professional Interface**
- Bootstrap 5 responsive design
- Font Awesome icons throughout
- Modern gradient color scheme
- Clean, intuitive navigation

### **Data Intelligence**
- Smart data interpretation (e.g., Sex: 0→Female, 1→Male)
- Summary statistics for predictions
- Performance trend analysis
- Model comparison capabilities

### **Error Resilience**
- Comprehensive error handling
- Graceful degradation when files are missing
- User-friendly error messages
- Safe navigation back to working areas

## 🔒 Security Notes

- **Development Server**: Current setup uses Flask development server
- **Production Deployment**: Use proper WSGI server (gunicorn, uwsgi) for production
- **Access Control**: No authentication implemented - add as needed
- **File Security**: Direct file serving - consider access restrictions

## 🚀 Production Deployment

For production use, consider:

1. **WSGI Server**:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 web_ui.app:app
```

2. **Reverse Proxy**: Use nginx for static files and SSL
3. **Authentication**: Add user login if needed
4. **Monitoring**: Add logging and health checks

## 🛠️ Customization

### **Styling**
- Edit CSS in `templates/base.html`
- Modify color schemes in the `<style>` section
- Add custom components as needed

### **Features**
- Add new routes in `app.py`
- Create corresponding templates
- Extend data processing functions

### **Branding**
- Update sidebar title and logo
- Customize color gradients
- Add company-specific elements

## 📈 Usage Examples

### **Viewing Predictions**
1. Navigate to "Predictions" in sidebar
2. Click "View" on any prediction file
3. See detailed results with statistics
4. Download CSV for external analysis

### **Monitoring Training**
1. Go to "Training History"
2. View all training sessions with scores
3. Click "Details" for comprehensive information
4. Compare different feature combinations

### **System Health**
1. Check "System Status" for health overview
2. Monitor file integrity and recent activity
3. Verify all components are functioning

This web UI provides a complete, professional interface for managing and viewing your ML pipeline results with an intuitive, modern design.
