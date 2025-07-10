# 🌍 PLSTM Land Subsidence Detection System

A modern, responsive web application for land subsidence prediction using Parallel Long Short-Term Memory (PLSTM) architecture with RINEX data processing.

## 🚀 Features

### 📊 **Complete ML Pipeline**
- **6-Step Training Workflow**: From data upload to model download
- **Real-time Progress Tracking**: Live training metrics and visualizations
- **Data Quality Assessment**: Missing values analysis and outlier detection
- **Interactive Data Preview**: Sample data tables and statistical summaries

### 🧠 **PLSTM Model Training**
- **Advanced Architecture Configuration**: LSTM units, layers, dropout rates
- **Real-time Training Dashboard**: Live loss curves and performance metrics
- **Model Export Options**: Complete model, weights, configuration, and reports
- **Comprehensive Evaluation**: RMSE, MAE, MSE metrics

### 🔮 **Real-time Prediction**
- **Instant Subsidence Predictions**: Input coordinates for immediate results
- **Risk Assessment**: Visual risk gauge with color-coded levels
- **Confidence Metrics**: Prediction reliability indicators
- **Trend Analysis**: Subsidence pattern insights

### 📈 **Advanced Analysis Tools**
- **Multiple Analysis Types**: Temporal, spatial, correlation, and trend analysis
- **Interactive Charts**: Time series and spatial distribution visualizations
- **Configurable Time Ranges**: Filter by year or view complete dataset
- **Dynamic Updates**: Real-time chart refreshing

### 🗺️ **Interactive Risk Mapping**
- **Leaflet Integration**: Interactive map of Padang City
- **Risk Visualization**: Color-coded subsidence risk areas
- **Detailed Popups**: Location-specific subsidence information
- **Layer Controls**: Switch between risk levels and time periods

## 🛠️ Technology Stack

### **Frontend**
- **HTML5**: Semantic markup with modern structure
- **CSS3**: Advanced styling with CSS Grid, Flexbox, and custom properties
- **JavaScript (ES6+)**: Modern JavaScript with async/await and modules
- **Chart.js**: Interactive data visualizations
- **Leaflet**: Interactive mapping functionality
- **Font Awesome**: Professional icon library

### **Design System**
- **Modern UI/UX**: Clean, professional interface design
- **Responsive Layout**: Mobile-first design approach
- **CSS Variables**: Consistent design tokens
- **Smooth Animations**: Micro-interactions and transitions
- **Accessibility**: WCAG compliant design patterns

## 📁 Project Structure

```
plstm-land-subsidence/
├── index.html              # Main application entry point
├── styles.css              # Complete styling system
├── script.js               # Application logic and functionality
├── README.md               # Project documentation
└── assets/                 # Static assets (if any)
```

## 🚀 Getting Started

### **Local Development**

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd plstm-land-subsidence
   ```

2. **Serve the application**
   
   **Option A: Using Python**
   ```bash
   # Python 3
   python -m http.server 8000
   
   # Python 2
   python -M SimpleHTTPServer 8000
   ```
   
   **Option B: Using Node.js**
   ```bash
   npx serve .
   ```
   
   **Option C: Using PHP**
   ```bash
   php -S localhost:8000
   ```

3. **Open in browser**
   ```
   http://localhost:8000
   ```

### **Netlify Deployment**

1. **Connect Repository**
   - Link your GitHub/GitLab repository to Netlify
   - Or drag and drop the project folder to Netlify

2. **Build Settings**
   - **Build command**: (leave empty - static site)
   - **Publish directory**: `/` (root directory)

3. **Deploy**
   - Automatic deployment on every push
   - Custom domain configuration available

## 📊 Data Processing Pipeline

```
Raw GNSS Data (.sth) → RINEX Conversion → Leica Infinity Processing → 
Clean CSV Data → Feature Engineering → PLSTM Training → Model Deployment
```

### **Supported Data Formats**
- **RINEX CSV Files**: Processed coordinate data from Leica Infinity
- **Required Columns**: Point ID, Timestamp, Coordinates, Heights
- **File Naming**: `cpdg2021.csv`, `cpdg2022.csv`, etc.

## 🔬 PLSTM Architecture

### **Model Configuration**
- **LSTM Units**: 32-256 (configurable)
- **Layers**: 1-4 parallel LSTM layers
- **Dropout Rate**: 0.0-0.5 for regularization
- **Sequence Length**: 5-50 timesteps
- **Prediction Horizon**: 1-10 future steps

### **Training Parameters**
- **Epochs**: 10-200 training iterations
- **Batch Size**: 16, 32, 64, 128
- **Learning Rate**: 0.0001, 0.001, 0.01
- **Data Split**: Configurable train/validation/test ratios

### **Evaluation Metrics**
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error
- **Temporal Accuracy**: Time-based prediction precision

## 🎯 Research Objectives

1. **PLSTM Model Design**: Effective time-series data handling for multiple sub-regions
2. **RINEX Data Integration**: High-precision GPS measurements for accurate predictions
3. **Performance Evaluation**: Comparison with traditional ML techniques
4. **Risk Management**: ML and regulatory policy combination strategies

## 📱 Responsive Design

### **Breakpoints**
- **Desktop**: 1024px and above
- **Tablet**: 768px - 1023px
- **Mobile**: 320px - 767px

### **Features**
- **Flexible Grid Layouts**: Automatic column adjustment
- **Touch-Friendly Interface**: Optimized for mobile interaction
- **Scalable Typography**: Responsive font sizing
- **Adaptive Navigation**: Collapsible menu for smaller screens

## 🎨 Design Features

### **Modern Aesthetics**
- **Gradient Backgrounds**: Professional color schemes
- **Card-Based Layout**: Clean content organization
- **Smooth Animations**: Engaging micro-interactions
- **Consistent Spacing**: 8px grid system

### **Color System**
- **Primary**: #667eea (Blue)
- **Secondary**: #764ba2 (Purple)
- **Accent**: #f093fb (Pink)
- **Success**: #4ade80 (Green)
- **Warning**: #fbbf24 (Yellow)
- **Error**: #ef4444 (Red)

## 🔧 Customization

### **Configuration Options**
- **Model Parameters**: Easily adjustable via UI sliders
- **Data Processing**: Configurable preprocessing options
- **Visualization**: Customizable chart types and colors
- **Analysis Types**: Multiple analysis modes available

### **Extending Functionality**
- **New Analysis Types**: Add custom analysis functions
- **Additional Metrics**: Implement new evaluation methods
- **Custom Visualizations**: Create specialized charts
- **API Integration**: Connect to external data sources

## 📈 Performance Optimization

### **Frontend Optimization**
- **Lazy Loading**: Charts and maps load on demand
- **Efficient Animations**: CSS transforms and GPU acceleration
- **Optimized Assets**: Compressed images and minified code
- **Caching Strategy**: Browser caching for static assets

### **User Experience**
- **Progressive Loading**: Step-by-step workflow
- **Real-time Feedback**: Live progress indicators
- **Error Handling**: Graceful error messages
- **Accessibility**: Keyboard navigation and screen reader support

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/new-feature`
3. **Commit changes**: `git commit -am 'Add new feature'`
4. **Push to branch**: `git push origin feature/new-feature`
5. **Submit a Pull Request**

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Research Team**: Land subsidence prediction research
- **Data Sources**: RINEX GPS measurement data
- **Open Source Libraries**: Chart.js, Leaflet, Font Awesome
- **Design Inspiration**: Modern web application patterns

## 📞 Support

For questions, issues, or contributions:
- **Create an Issue**: Use GitHub issues for bug reports
- **Documentation**: Check README and inline comments
- **Community**: Join discussions in project forums

---

**© 2025 - PLSTM Land Subsidence Detection System | Advanced ML Research**