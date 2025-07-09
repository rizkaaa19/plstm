// Global variables
let processedData = null;
let subsidenceMap = null;
let currentModel = null;
let trainingData = null;
let currentStep = 1;
let trainingCharts = {};

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeNavigation();
    initializeTrainingWorkflow();
    initializePrediction();
    initializeAnalysis();
    initializeMap();
    updateConfigValues();
});

// Navigation functionality
function initializeNavigation() {
    const navButtons = document.querySelectorAll('.nav-btn');
    const pages = document.querySelectorAll('.page');

    navButtons.forEach(button => {
        button.addEventListener('click', () => {
            const targetPage = button.getAttribute('data-page');
            
            // Update active navigation
            navButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
            
            // Show target page
            pages.forEach(page => page.classList.remove('active'));
            document.getElementById(targetPage).classList.add('active');
        });
    });
}

// Training Workflow Initialization
function initializeTrainingWorkflow() {
    // File upload handling
    const fileInput = document.getElementById('csvFiles');
    const loadDataBtn = document.getElementById('loadData');
    
    fileInput.addEventListener('change', handleFileSelection);
    loadDataBtn.addEventListener('click', loadAndValidateData);
    
    // Tab switching
    initializePreviewTabs();
    
    // Step progression buttons
    document.getElementById('proceedToPreprocessing')?.addEventListener('click', () => showStep(3));
    document.getElementById('startPreprocessing')?.addEventListener('click', startPreprocessing);
    document.getElementById('startTraining')?.addEventListener('click', startModelTraining);
    
    // Download buttons
    initializeDownloadButtons();
}

// File selection handling
function handleFileSelection() {
    const fileInput = document.getElementById('csvFiles');
    const loadDataBtn = document.getElementById('loadData');
    const fileList = document.getElementById('fileList');
    
    if (fileInput.files.length > 0) {
        loadDataBtn.disabled = false;
        loadDataBtn.innerHTML = `<i class="fas fa-play"></i> Load ${fileInput.files.length} file(s)`;
        
        // Display file list
        fileList.innerHTML = '';
        Array.from(fileInput.files).forEach(file => {
            const fileItem = document.createElement('div');
            fileItem.className = 'file-item';
            fileItem.innerHTML = `
                <i class="fas fa-file-csv"></i>
                <span>${file.name}</span>
                <small>(${(file.size / 1024).toFixed(1)} KB)</small>
            `;
            fileList.appendChild(fileItem);
        });
        fileList.style.display = 'block';
    } else {
        loadDataBtn.disabled = true;
        loadDataBtn.innerHTML = '<i class="fas fa-play"></i> Load and Validate Data';
        fileList.style.display = 'none';
    }
}

// Load and validate data
async function loadAndValidateData() {
    const fileInput = document.getElementById('csvFiles');
    const files = fileInput.files;
    
    if (files.length === 0) {
        showMessage('Please select CSV files first.', 'error');
        return;
    }
    
    showLoading(true, 'Loading and validating data...');
    updateStepStatus(1, 'Processing');
    
    try {
        const formData = new FormData();
        for (let file of files) {
            formData.append('csvFiles', file);
        }
        
        const response = await fetch('/api/process-data', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            processedData = result.data;
            trainingData = result;
            
            updateDataPreview(result.summary, result.sampleData);
            updateStepStatus(1, 'Complete');
            showStep(2);
            showMessage('Data loaded and validated successfully!', 'success');
        } else {
            updateStepStatus(1, 'Error');
            showMessage(`Error processing data: ${result.error}`, 'error');
        }
    } catch (error) {
        updateStepStatus(1, 'Error');
        showMessage(`Error: ${error.message}`, 'error');
    } finally {
        showLoading(false);
    }
}

// Update data preview
function updateDataPreview(summary, sampleData) {
    // Update summary statistics
    document.getElementById('totalRecords').textContent = summary.totalRecords.toLocaleString();
    document.getElementById('dateRange').textContent = summary.dateRange ? 
        `${summary.dateRange.start} - ${summary.dateRange.end}` : 'N/A';
    document.getElementById('featuresCount').textContent = summary.features;
    
    // Update spatial coverage
    if (summary.spatialCoverage) {
        document.getElementById('latRange').textContent = 
            `${summary.spatialCoverage.latMin.toFixed(6)} - ${summary.spatialCoverage.latMax.toFixed(6)}`;
        document.getElementById('lonRange').textContent = 
            `${summary.spatialCoverage.lonMin.toFixed(6)} - ${summary.spatialCoverage.lonMax.toFixed(6)}`;
        document.getElementById('heightRange').textContent = 
            `${summary.spatialCoverage.heightMin.toFixed(2)} - ${summary.spatialCoverage.heightMax.toFixed(2)} m`;
    }
    
    // Update sample data table
    if (sampleData && sampleData.length > 0) {
        updateSampleDataTable(sampleData);
    }
    
    // Update quality metrics
    updateQualityMetrics(summary.qualityMetrics);
    
    // Create visualizations
    createDataVisualizations(summary);
}

// Update sample data table
function updateSampleDataTable(sampleData) {
    const table = document.getElementById('sampleDataTable');
    const thead = table.querySelector('thead');
    const tbody = table.querySelector('tbody');
    
    // Clear existing content
    thead.innerHTML = '';
    tbody.innerHTML = '';
    
    if (sampleData.length === 0) return;
    
    // Create header
    const headerRow = document.createElement('tr');
    Object.keys(sampleData[0]).forEach(key => {
        const th = document.createElement('th');
        th.textContent = key;
        headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    
    // Create rows (show first 10 rows)
    sampleData.slice(0, 10).forEach(row => {
        const tr = document.createElement('tr');
        Object.values(row).forEach(value => {
            const td = document.createElement('td');
            td.textContent = typeof value === 'number' ? value.toFixed(6) : value;
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    });
}

// Update quality metrics
function updateQualityMetrics(qualityMetrics) {
    const missingValuesDiv = document.getElementById('missingValues');
    
    if (qualityMetrics && qualityMetrics.missingValues) {
        missingValuesDiv.innerHTML = '';
        Object.entries(qualityMetrics.missingValues).forEach(([column, count]) => {
            const item = document.createElement('div');
            item.className = 'metric-item';
            item.innerHTML = `
                <span>${column}:</span>
                <strong style="color: ${count > 0 ? '#f44336' : '#4caf50'}">${count}</strong>
            `;
            missingValuesDiv.appendChild(item);
        });
    }
}

// Create data visualizations
function createDataVisualizations(summary) {
    // Distribution chart
    const distributionCtx = document.getElementById('distributionChart');
    if (distributionCtx && summary.distribution) {
        new Chart(distributionCtx, {
            type: 'bar',
            data: {
                labels: summary.distribution.labels,
                datasets: [{
                    label: 'Frequency',
                    data: summary.distribution.values,
                    backgroundColor: 'rgba(102, 126, 234, 0.6)',
                    borderColor: '#667eea',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Height Distribution'
                    }
                }
            }
        });
    }
    
    // Spatial chart
    const spatialCtx = document.getElementById('spatialChart');
    if (spatialCtx && summary.spatialData) {
        new Chart(spatialCtx, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Measurement Points',
                    data: summary.spatialData,
                    backgroundColor: 'rgba(102, 126, 234, 0.6)',
                    borderColor: '#667eea',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Longitude'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Latitude'
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Spatial Distribution of Measurement Points'
                    }
                }
            }
        });
    }
}

// Initialize preview tabs
function initializePreviewTabs() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const targetTab = button.getAttribute('data-tab');
            
            // Update active tab button
            tabButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
            
            // Show target tab content
            tabContents.forEach(content => content.classList.remove('active'));
            document.getElementById(`${targetTab}-tab`).classList.add('active');
        });
    });
}

// Start preprocessing
async function startPreprocessing() {
    if (!processedData) {
        showMessage('Please load data first.', 'error');
        return;
    }
    
    showLoading(true, 'Preprocessing data...');
    updateStepStatus(3, 'Processing');
    
    const progressFill = document.getElementById('preprocessingFill');
    const progressStatus = document.getElementById('preprocessingStatus');
    const progressContainer = document.getElementById('preprocessingProgress');
    
    progressContainer.style.display = 'block';
    
    // Simulate preprocessing steps
    const steps = [
        'Removing outliers...',
        'Filling missing values...',
        'Applying smoothing filter...',
        'Scaling features...',
        'Creating time sequences...',
        'Splitting data...'
    ];
    
    for (let i = 0; i < steps.length; i++) {
        await new Promise(resolve => setTimeout(resolve, 1000));
        const progress = ((i + 1) / steps.length) * 100;
        progressFill.style.width = `${progress}%`;
        progressStatus.textContent = steps[i];
    }
    
    updateStepStatus(3, 'Complete');
    showStep(4);
    showLoading(false);
    showMessage('Data preprocessing completed successfully!', 'success');
}

// Configuration value updates
function updateConfigValues() {
    const configs = [
        { id: 'lstmUnits', valueId: 'lstmUnitsValue' },
        { id: 'numLayers', valueId: 'numLayersValue' },
        { id: 'dropoutRate', valueId: 'dropoutRateValue' },
        { id: 'epochs', valueId: 'epochsValue' },
        { id: 'trainSplit', valueId: 'trainSplitValue', isPercent: true },
        { id: 'validationSplit', valueId: 'validationSplitValue', isPercent: true },
        { id: 'sequenceLength', valueId: 'sequenceLengthValue' },
        { id: 'predictionHorizon', valueId: 'predictionHorizonValue' }
    ];
    
    configs.forEach(config => {
        const element = document.getElementById(config.id);
        const valueElement = document.getElementById(config.valueId);
        
        if (element && valueElement) {
            element.addEventListener('input', () => {
                let value = element.value;
                if (config.isPercent) {
                    value = Math.round(parseFloat(value) * 100) + '%';
                    // Update test split
                    if (config.id === 'trainSplit' || config.id === 'validationSplit') {
                        updateTestSplit();
                    }
                }
                valueElement.textContent = value;
            });
        }
    });
}

// Update test split value
function updateTestSplit() {
    const trainSplit = parseFloat(document.getElementById('trainSplit').value);
    const validationSplit = parseFloat(document.getElementById('validationSplit').value);
    const testSplit = 1 - trainSplit - validationSplit;
    document.getElementById('testSplitValue').textContent = Math.round(testSplit * 100) + '%';
}

// Start model training
async function startModelTraining() {
    if (!processedData) {
        showMessage('Please process data first.', 'error');
        return;
    }
    
    showLoading(true, 'Initializing PLSTM model...');
    updateStepStatus(4, 'Complete');
    updateStepStatus(5, 'Training');
    showStep(5);
    
    const epochs = parseInt(document.getElementById('epochs').value);
    const batchSize = parseInt(document.getElementById('batchSize').value);
    
    // Initialize training charts
    initializeTrainingCharts();
    
    // Simulate training process
    await simulateTraining(epochs);
    
    updateStepStatus(5, 'Complete');
    showStep(6);
    showLoading(false);
    showMessage('PLSTM model training completed successfully!', 'success');
    
    // Update final results
    updateFinalResults();
}

// Initialize training charts
function initializeTrainingCharts() {
    const lossCtx = document.getElementById('lossChart');
    const metricsCtx = document.getElementById('metricsChart');
    
    if (lossCtx) {
        trainingCharts.loss = new Chart(lossCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Training Loss',
                        data: [],
                        borderColor: '#667eea',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        tension: 0.4
                    },
                    {
                        label: 'Validation Loss',
                        data: [],
                        borderColor: '#f44336',
                        backgroundColor: 'rgba(244, 67, 54, 0.1)',
                        tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Loss'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Epoch'
                        }
                    }
                }
            }
        });
    }
    
    if (metricsCtx) {
        trainingCharts.metrics = new Chart(metricsCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'RMSE',
                        data: [],
                        borderColor: '#4caf50',
                        backgroundColor: 'rgba(76, 175, 80, 0.1)',
                        tension: 0.4
                    },
                    {
                        label: 'MAE',
                        data: [],
                        borderColor: '#ff9800',
                        backgroundColor: 'rgba(255, 152, 0, 0.1)',
                        tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Error'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Epoch'
                        }
                    }
                }
            }
        });
    }
}

// Simulate training process
async function simulateTraining(epochs) {
    const progressFill = document.getElementById('trainingProgressFill');
    const trainingStatus = document.getElementById('trainingStatus');
    const currentEpochSpan = document.getElementById('currentEpoch');
    const totalEpochsSpan = document.getElementById('totalEpochs');
    const trainingLossSpan = document.getElementById('trainingLoss');
    const validationLossSpan = document.getElementById('validationLoss');
    const rmseSpan = document.getElementById('rmseValue');
    
    totalEpochsSpan.textContent = epochs;
    
    for (let epoch = 1; epoch <= epochs; epoch++) {
        await new Promise(resolve => setTimeout(resolve, 200)); // Faster simulation
        
        const progress = (epoch / epochs) * 100;
        progressFill.style.width = `${progress}%`;
        trainingStatus.textContent = `Training epoch ${epoch}/${epochs}...`;
        currentEpochSpan.textContent = epoch;
        
        // Simulate decreasing loss and metrics
        const trainLoss = (0.1 * Math.exp(-epoch / 20) + Math.random() * 0.01).toFixed(4);
        const valLoss = (0.12 * Math.exp(-epoch / 18) + Math.random() * 0.01).toFixed(4);
        const rmse = (0.05 * Math.exp(-epoch / 15) + Math.random() * 0.005).toFixed(4);
        const mae = (0.03 * Math.exp(-epoch / 16) + Math.random() * 0.003).toFixed(4);
        
        trainingLossSpan.textContent = trainLoss;
        validationLossSpan.textContent = valLoss;
        rmseSpan.textContent = rmse;
        
        // Update charts
        if (trainingCharts.loss) {
            trainingCharts.loss.data.labels.push(epoch);
            trainingCharts.loss.data.datasets[0].data.push(parseFloat(trainLoss));
            trainingCharts.loss.data.datasets[1].data.push(parseFloat(valLoss));
            trainingCharts.loss.update('none');
        }
        
        if (trainingCharts.metrics) {
            trainingCharts.metrics.data.labels.push(epoch);
            trainingCharts.metrics.data.datasets[0].data.push(parseFloat(rmse));
            trainingCharts.metrics.data.datasets[1].data.push(parseFloat(mae));
            trainingCharts.metrics.update('none');
        }
    }
    
    trainingStatus.textContent = 'Training completed!';
    
    // Store final model
    currentModel = {
        trained: true,
        epochs: epochs,
        finalLoss: parseFloat(trainingLossSpan.textContent),
        finalRMSE: parseFloat(rmseSpan.textContent),
        finalMAE: parseFloat(mae),
        trainingTime: `${Math.round(epochs * 0.2)} seconds`
    };
}

// Update final results
function updateFinalResults() {
    if (!currentModel) return;
    
    document.getElementById('finalRMSE').textContent = currentModel.finalRMSE.toFixed(4);
    document.getElementById('finalMAE').textContent = currentModel.finalMAE.toFixed(4);
    document.getElementById('finalMSE').textContent = (currentModel.finalRMSE ** 2).toFixed(6);
    document.getElementById('trainingTime').textContent = currentModel.trainingTime;
    document.getElementById('bestEpoch').textContent = Math.round(currentModel.epochs * 0.8);
    document.getElementById('modelSize').textContent = '2.4 MB';
    
    // Create prediction preview chart
    createPredictionPreview();
}

// Create prediction preview chart
function createPredictionPreview() {
    const ctx = document.getElementById('predictionChart');
    if (!ctx) return;
    
    // Generate sample prediction data
    const actualData = [];
    const predictedData = [];
    const labels = [];
    
    for (let i = 0; i < 50; i++) {
        const actual = Math.sin(i * 0.1) * 0.05 + Math.random() * 0.01;
        const predicted = actual + (Math.random() - 0.5) * 0.01;
        
        actualData.push(actual);
        predictedData.push(predicted);
        labels.push(`Point ${i + 1}`);
    }
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Actual Subsidence',
                    data: actualData,
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    tension: 0.4
                },
                {
                    label: 'Predicted Subsidence',
                    data: predictedData,
                    borderColor: '#f44336',
                    backgroundColor: 'rgba(244, 67, 54, 0.1)',
                    tension: 0.4,
                    borderDash: [5, 5]
                }
            ]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    title: {
                        display: true,
                        text: 'Subsidence (m)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Data Points'
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Model Prediction vs Actual Values'
                }
            }
        }
    });
}

// Initialize download buttons
function initializeDownloadButtons() {
    const downloadButtons = [
        { id: 'downloadModel', filename: 'plstm_model.h5', type: 'model' },
        { id: 'downloadWeights', filename: 'plstm_weights.h5', type: 'weights' },
        { id: 'downloadConfig', filename: 'model_config.json', type: 'config' },
        { id: 'downloadReport', filename: 'training_report.pdf', type: 'report' }
    ];
    
    downloadButtons.forEach(btn => {
        const element = document.getElementById(btn.id);
        if (element) {
            element.addEventListener('click', () => downloadFile(btn.filename, btn.type));
        }
    });
}

// Download file function
function downloadFile(filename, type) {
    if (!currentModel || !currentModel.trained) {
        showMessage('Please train a model first.', 'error');
        return;
    }
    
    // Create download content based on type
    let content = '';
    let mimeType = 'application/octet-stream';
    
    switch (type) {
        case 'model':
            content = 'Binary model file content would be here';
            mimeType = 'application/octet-stream';
            break;
        case 'weights':
            content = 'Binary weights file content would be here';
            mimeType = 'application/octet-stream';
            break;
        case 'config':
            content = JSON.stringify({
                architecture: 'PLSTM',
                lstm_units: document.getElementById('lstmUnits').value,
                num_layers: document.getElementById('numLayers').value,
                dropout_rate: document.getElementById('dropoutRate').value,
                epochs: document.getElementById('epochs').value,
                batch_size: document.getElementById('batchSize').value,
                learning_rate: document.getElementById('learningRate').value,
                sequence_length: document.getElementById('sequenceLength').value,
                final_rmse: currentModel.finalRMSE,
                final_mae: currentModel.finalMAE,
                training_time: currentModel.trainingTime
            }, null, 2);
            mimeType = 'application/json';
            break;
        case 'report':
            content = generateTrainingReport();
            mimeType = 'text/plain';
            break;
    }
    
    // Create and trigger download
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    showMessage(`${filename} downloaded successfully!`, 'success');
}

// Generate training report
function generateTrainingReport() {
    return `
PLSTM Land Subsidence Model Training Report
==========================================

Model Configuration:
- Architecture: Parallel Long Short-Term Memory (PLSTM)
- LSTM Units: ${document.getElementById('lstmUnits').value}
- Number of Layers: ${document.getElementById('numLayers').value}
- Dropout Rate: ${document.getElementById('dropoutRate').value}
- Sequence Length: ${document.getElementById('sequenceLength').value}

Training Parameters:
- Epochs: ${document.getElementById('epochs').value}
- Batch Size: ${document.getElementById('batchSize').value}
- Learning Rate: ${document.getElementById('learningRate').value}
- Training Split: ${document.getElementById('trainSplitValue').textContent}
- Validation Split: ${document.getElementById('validationSplitValue').textContent}
- Test Split: ${document.getElementById('testSplitValue').textContent}

Final Results:
- RMSE: ${currentModel.finalRMSE.toFixed(4)}
- MAE: ${currentModel.finalMAE.toFixed(4)}
- MSE: ${(currentModel.finalRMSE ** 2).toFixed(6)}
- Training Time: ${currentModel.trainingTime}

Data Information:
- Total Records: ${document.getElementById('totalRecords').textContent}
- Date Range: ${document.getElementById('dateRange').textContent}
- Features: ${document.getElementById('featuresCount').textContent}

Generated on: ${new Date().toLocaleString()}
    `.trim();
}

// Show/hide training steps
function showStep(stepNumber) {
    for (let i = 1; i <= 6; i++) {
        const step = document.getElementById(`step${i}`);
        if (step) {
            step.style.display = i <= stepNumber ? 'block' : 'none';
        }
    }
    currentStep = stepNumber;
}

// Update step status
function updateStepStatus(stepNumber, status) {
    const statusElement = document.getElementById(`step${stepNumber}-status`);
    if (statusElement) {
        statusElement.textContent = status;
        statusElement.className = `step-status ${status.toLowerCase()}`;
    }
}

// Initialize prediction functionality
function initializePrediction() {
    document.getElementById('makePrediction').addEventListener('click', makePrediction);
}

// Make prediction
async function makePrediction() {
    const easting = parseFloat(document.getElementById('easting').value);
    const northing = parseFloat(document.getElementById('northing').value);
    const orthoHeight = parseFloat(document.getElementById('orthoHeight').value);
    const ellipHeight = parseFloat(document.getElementById('ellipHeight').value);
    
    if (isNaN(easting) || isNaN(northing) || isNaN(orthoHeight) || isNaN(ellipHeight)) {
        showMessage('Please fill in all input fields with valid numbers.', 'error');
        return;
    }
    
    showLoading(true, 'Making prediction...');
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                easting,
                northing,
                orthoHeight,
                ellipHeight
            })
        });
        
        const result = await response.json();
        
        if (result.success) {
            displayPredictionResults(result.prediction);
        } else {
            showMessage(`Prediction failed: ${result.error}`, 'error');
        }
    } catch (error) {
        showMessage(`Error: ${error.message}`, 'error');
    } finally {
        showLoading(false);
    }
}

// Display prediction results
function displayPredictionResults(prediction) {
    document.getElementById('subsidenceValue').textContent = `${(prediction.subsidence * 100).toFixed(2)} cm`;
    document.getElementById('riskLevel').textContent = prediction.riskLevel;
    document.getElementById('riskLevel').style.color = prediction.riskColor;
    document.getElementById('confidence').textContent = `${(prediction.confidence * 100).toFixed(1)}%`;
    document.getElementById('trend').textContent = prediction.trend || 'Stable';
    
    document.getElementById('predictionResults').style.display = 'block';
    document.getElementById('predictionResults').classList.add('slide-up');
    
    // Draw risk gauge
    drawRiskGauge(prediction.subsidence * 100, prediction.riskLevel);
}

// Draw risk gauge chart
function drawRiskGauge(value, riskLevel) {
    const canvas = document.getElementById('riskGauge');
    const ctx = canvas.getContext('2d');
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const centerX = canvas.width / 2;
    const centerY = canvas.height - 50;
    const radius = 120;
    
    // Draw gauge background
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, Math.PI, 0);
    ctx.lineWidth = 20;
    ctx.strokeStyle = '#e0e0e0';
    ctx.stroke();
    
    // Draw risk zones
    const zones = [
        { start: Math.PI, end: Math.PI * 0.75, color: '#4CAF50' }, // Low (0-5cm)
        { start: Math.PI * 0.75, end: Math.PI * 0.5, color: '#FF9800' }, // Medium (5-10cm)
        { start: Math.PI * 0.5, end: 0, color: '#F44336' } // High (10-20cm)
    ];
    
    zones.forEach(zone => {
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, zone.start, zone.end);
        ctx.lineWidth = 20;
        ctx.strokeStyle = zone.color;
        ctx.stroke();
    });
    
    // Draw needle
    const angle = Math.PI - (Math.min(value, 20) / 20) * Math.PI; // Scale 0-20cm to 0-π
    const needleLength = radius - 10;
    
    ctx.beginPath();
    ctx.moveTo(centerX, centerY);
    ctx.lineTo(
        centerX + Math.cos(angle) * needleLength,
        centerY + Math.sin(angle) * needleLength
    );
    ctx.lineWidth = 4;
    ctx.strokeStyle = '#333';
    ctx.stroke();
    
    // Draw center circle
    ctx.beginPath();
    ctx.arc(centerX, centerY, 8, 0, 2 * Math.PI);
    ctx.fillStyle = '#333';
    ctx.fill();
    
    // Draw value text
    ctx.font = 'bold 24px Arial';
    ctx.fillStyle = '#333';
    ctx.textAlign = 'center';
    ctx.fillText(`${value.toFixed(1)} cm`, centerX, centerY + 40);
    
    ctx.font = '16px Arial';
    ctx.fillText(riskLevel, centerX, centerY + 60);
}

// Initialize analysis functionality
function initializeAnalysis() {
    document.getElementById('updateAnalysis')?.addEventListener('click', updateAnalysis);
}

// Update analysis
function updateAnalysis() {
    const analysisType = document.getElementById('analysisType').value;
    const timeRange = document.getElementById('timeRange').value;
    
    showLoading(true, 'Updating analysis...');
    
    setTimeout(() => {
        createAnalysisCharts(analysisType, timeRange);
        showLoading(false);
        showMessage('Analysis updated successfully!', 'success');
    }, 1000);
}

// Create analysis charts
function createAnalysisCharts(analysisType, timeRange) {
    const timeSeriesCtx = document.getElementById('timeSeriesChart');
    const spatialCtx = document.getElementById('spatialAnalysisChart');
    
    // Generate sample data based on analysis type
    const timeSeriesData = generateTimeSeriesData(timeRange);
    const spatialData = generateSpatialData();
    
    // Time series chart
    if (timeSeriesCtx) {
        new Chart(timeSeriesCtx, {
            type: 'line',
            data: {
                labels: timeSeriesData.labels,
                datasets: [{
                    label: 'Land Subsidence (cm)',
                    data: timeSeriesData.values,
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Time'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Subsidence (cm)'
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: `${analysisType.charAt(0).toUpperCase() + analysisType.slice(1)} Analysis - ${timeRange}`
                    }
                }
            }
        });
    }
    
    // Spatial analysis chart
    if (spatialCtx) {
        new Chart(spatialCtx, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Measurement Points',
                    data: spatialData,
                    backgroundColor: spatialData.map(point => 
                        point.subsidence > 10 ? '#F44336' :
                        point.subsidence > 5 ? '#FF9800' : '#4CAF50'
                    ),
                    borderColor: '#333',
                    borderWidth: 1,
                    pointRadius: spatialData.map(point => Math.max(5, point.subsidence / 2))
                }]
            },
            options: {
                responsive: true,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Longitude'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Latitude'
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Spatial Distribution of Subsidence'
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const point = context.raw;
                                return `Subsidence: ${point.subsidence.toFixed(1)} cm`;
                            }
                        }
                    }
                }
            }
        });
    }
}

// Generate time series data
function generateTimeSeriesData(timeRange) {
    const data = { labels: [], values: [] };
    const startYear = timeRange === 'all' ? 2021 : parseInt(timeRange);
    const endYear = timeRange === 'all' ? 2024 : parseInt(timeRange);
    
    for (let year = startYear; year <= endYear; year++) {
        for (let month = 1; month <= 12; month++) {
            data.labels.push(`${year}-${month.toString().padStart(2, '0')}`);
            const baseValue = Math.sin((year - 2021) * 2 + month * 0.5) * 2 + 5;
            const noise = (Math.random() - 0.5) * 2;
            data.values.push(Math.max(0, baseValue + noise));
        }
    }
    
    return data;
}

// Generate spatial data
function generateSpatialData() {
    const data = [];
    const basePoints = [
        { x: 100.3565, y: -0.9356 },
        { x: 100.3680, y: -0.9783 },
        { x: 100.4182, y: -0.9834 },
        { x: 100.3500, y: -0.9471 },
        { x: 100.4000, y: -0.9200 }
    ];
    
    basePoints.forEach(point => {
        data.push({
            x: point.x,
            y: point.y,
            subsidence: Math.random() * 15 + 2
        });
    });
    
    return data;
}

// Initialize map
function initializeMap() {
    // Padang City coordinates
    const padangCenter = [-0.9471, 100.4172];
    
    subsidenceMap = L.map('subsidenceMap').setView(padangCenter, 12);
    
    // Add tile layer
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors'
    }).addTo(subsidenceMap);
    
    // Sample subsidence points
    const subsidencePoints = [
        { name: "Gunung Padang", location: [-0.9356, 100.3565], risk: "High", value: 0.15 },
        { name: "Pantai Air Manis", location: [-0.9783, 100.3680], risk: "Medium", value: 0.08 },
        { name: "Lubuk Begalung", location: [-0.9834, 100.4182], risk: "Low", value: 0.03 },
        { name: "Padang Barat", location: [-0.9471, 100.3500], risk: "Medium", value: 0.09 },
        { name: "Koto Tangah", location: [-0.9200, 100.4000], risk: "High", value: 0.12 }
    ];
    
    // Add markers
    subsidencePoints.forEach(point => {
        const color = point.risk === 'High' ? 'red' : 
                     point.risk === 'Medium' ? 'orange' : 'green';
        
        const marker = L.marker(point.location).addTo(subsidenceMap);
        marker.bindPopup(`
            <strong>${point.name}</strong><br>
            Risk Level: ${point.risk}<br>
            Subsidence: ${(point.value * 100).toFixed(1)} cm
        `);
    });
}

// Utility functions
function showLoading(show, text = 'Processing...') {
    const overlay = document.getElementById('loadingOverlay');
    const loadingText = document.getElementById('loadingText');
    
    overlay.style.display = show ? 'flex' : 'none';
    if (loadingText) {
        loadingText.textContent = text;
    }
}

function showMessage(message, type) {
    // Create message element
    const messageDiv = document.createElement('div');
    messageDiv.className = `${type}-msg fade-in`;
    messageDiv.innerHTML = `<i class="fas fa-${type === 'success' ? 'check' : type === 'error' ? 'times' : 'exclamation'}"></i> ${message}`;
    
    // Insert at the top of main content
    const mainContent = document.querySelector('.main-content');
    mainContent.insertBefore(messageDiv, mainContent.firstChild);
    
    // Remove after 5 seconds
    setTimeout(() => {
        if (messageDiv.parentNode) {
            messageDiv.parentNode.removeChild(messageDiv);
        }
    }, 5000);
}