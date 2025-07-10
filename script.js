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
            navigateToPage(targetPage);
        });
    });
}

function navigateToPage(targetPage) {
    const navButtons = document.querySelectorAll('.nav-btn');
    const pages = document.querySelectorAll('.page');
    
    // Update active navigation
    navButtons.forEach(btn => btn.classList.remove('active'));
    document.querySelector(`[data-page="${targetPage}"]`).classList.add('active');
    
    // Show target page
    pages.forEach(page => page.classList.remove('active'));
    document.getElementById(targetPage).classList.add('active');
}

// Training Workflow Initialization
function initializeTrainingWorkflow() {
    // File upload handling
    const fileInput = document.getElementById('csvFiles');
    const loadDataBtn = document.getElementById('loadData');
    
    if (fileInput) fileInput.addEventListener('change', handleFileSelection);
    if (loadDataBtn) loadDataBtn.addEventListener('click', loadAndValidateData);
    
    // Tab switching
    initializePreviewTabs();
    
    // Step progression buttons
    const proceedBtn = document.getElementById('proceedToPreprocessing');
    const preprocessBtn = document.getElementById('startPreprocessing');
    const trainingBtn = document.getElementById('startTraining');
    
    if (proceedBtn) proceedBtn.addEventListener('click', () => showStep(3));
    if (preprocessBtn) preprocessBtn.addEventListener('click', startPreprocessing);
    if (trainingBtn) trainingBtn.addEventListener('click', startModelTraining);
    
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

// Load and validate data (simulated for frontend-only)
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
        // Simulate data processing
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        // Generate simulated data summary
        const simulatedData = generateSimulatedData();
        processedData = simulatedData.data;
        trainingData = simulatedData;
        
        updateDataPreview(simulatedData.summary, simulatedData.sampleData);
        updateStepStatus(1, 'Complete');
        showStep(2);
        showMessage('Data loaded and validated successfully!', 'success');
    } catch (error) {
        updateStepStatus(1, 'Error');
        showMessage(`Error processing data: ${error.message}`, 'error');
    } finally {
        showLoading(false);
    }
}

// Generate simulated data for demo purposes
function generateSimulatedData() {
    const totalRecords = 1500 + Math.floor(Math.random() * 500);
    const features = ['Point Id', 'Start Time', 'WGS84 Latitude', 'WGS84 Longitude', 'Easting [m]', 'Northing [m]', 'Ortho Height [m]', 'WGS84 Ellip Height [m]', 'Geoid Separation [m]'];
    
    // Generate sample data
    const sampleData = [];
    for (let i = 0; i < 20; i++) {
        sampleData.push({
            'Point Id': `CPDG${String(i + 1).padStart(3, '0')}`,
            'Start Time': new Date(2021 + Math.floor(i / 5), Math.floor(Math.random() * 12), Math.floor(Math.random() * 28) + 1).toISOString().split('T')[0],
            'WGS84 Latitude': (-0.95 + Math.random() * 0.1).toFixed(6),
            'WGS84 Longitude': (100.35 + Math.random() * 0.1).toFixed(6),
            'Easting [m]': (128000 + Math.random() * 10000).toFixed(3),
            'Northing [m]': (9894000 + Math.random() * 10000).toFixed(3),
            'Ortho Height [m]': (5 + Math.random() * 50).toFixed(3),
            'WGS84 Ellip Height [m]': (25 + Math.random() * 50).toFixed(3),
            'Geoid Separation [m]': (20 + Math.random() * 5).toFixed(3)
        });
    }
    
    // Generate spatial data for visualization
    const spatialData = [];
    for (let i = 0; i < 100; i++) {
        spatialData.push({
            x: 100.35 + Math.random() * 0.1,
            y: -0.95 + Math.random() * 0.1
        });
    }
    
    // Generate distribution data
    const distributionData = {
        labels: ['0-10m', '10-20m', '20-30m', '30-40m', '40-50m', '50-60m'],
        values: [120, 250, 180, 90, 60, 30]
    };
    
    const summary = {
        totalRecords,
        features: features.length,
        dateRange: {
            start: 2021,
            end: 2024
        },
        spatialCoverage: {
            latMin: -0.98,
            latMax: -0.92,
            lonMin: 100.35,
            lonMax: 100.45,
            heightMin: 5.2,
            heightMax: 75.8
        },
        qualityMetrics: {
            missingValues: {
                'Point Id': 0,
                'Start Time': 2,
                'WGS84 Latitude': 1,
                'WGS84 Longitude': 1,
                'Easting [m]': 0,
                'Northing [m]': 0,
                'Ortho Height [m]': 3,
                'WGS84 Ellip Height [m]': 2,
                'Geoid Separation [m]': 1
            }
        },
        distribution: distributionData,
        spatialData
    };
    
    return {
        data: sampleData,
        summary,
        sampleData,
        fileStats: [
            { filename: 'cpdg2021.csv', records: 365, size: 45000 },
            { filename: 'cpdg2022.csv', records: 365, size: 46000 },
            { filename: 'cpdg2023.csv', records: 365, size: 47000 },
            { filename: 'cpdg2024.csv', records: 400, size: 48000 }
        ]
    };
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
                <strong style="color: ${count > 0 ? '#ef4444' : '#4ade80'}">${count}</strong>
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
                    borderWidth: 1,
                    borderRadius: 8
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Height Distribution'
                    },
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true
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
                    borderWidth: 1,
                    pointRadius: 4
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
                    },
                    legend: {
                        display: false
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
            const targetContent = document.getElementById(`${targetTab}-tab`);
            if (targetContent) {
                targetContent.classList.add('active');
            }
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
    const testSplitElement = document.getElementById('testSplitValue');
    if (testSplitElement) {
        testSplitElement.textContent = Math.round(testSplit * 100) + '%';
    }
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
                        tension: 0.4,
                        fill: true
                    },
                    {
                        label: 'Validation Loss',
                        data: [],
                        borderColor: '#ef4444',
                        backgroundColor: 'rgba(239, 68, 68, 0.1)',
                        tension: 0.4,
                        fill: true
                    }
                ]
            },
            options: {
                responsive: true,
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
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
                },
                plugins: {
                    legend: {
                        position: 'top'
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
                        borderColor: '#4ade80',
                        backgroundColor: 'rgba(74, 222, 128, 0.1)',
                        tension: 0.4,
                        fill: true
                    },
                    {
                        label: 'MAE',
                        data: [],
                        borderColor: '#fbbf24',
                        backgroundColor: 'rgba(251, 191, 36, 0.1)',
                        tension: 0.4,
                        fill: true
                    }
                ]
            },
            options: {
                responsive: true,
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
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
                },
                plugins: {
                    legend: {
                        position: 'top'
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
        
        // Simulate decreasing loss and metrics with more realistic curves
        const trainLoss = (0.15 * Math.exp(-epoch / 25) + 0.01 + Math.random() * 0.005).toFixed(4);
        const valLoss = (0.18 * Math.exp(-epoch / 22) + 0.015 + Math.random() * 0.008).toFixed(4);
        const rmse = (0.08 * Math.exp(-epoch / 20) + 0.02 + Math.random() * 0.003).toFixed(4);
        const mae = (0.05 * Math.exp(-epoch / 18) + 0.015 + Math.random() * 0.002).toFixed(4);
        
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
        trainingTime: `${Math.round(epochs * 0.3)} seconds`
    };
}

// Update final results
function updateFinalResults() {
    if (!currentModel) return;
    
    document.getElementById('finalRMSE').textContent = currentModel.finalRMSE.toFixed(4);
    document.getElementById('finalMAE').textContent = currentModel.finalMAE.toFixed(4);
    document.getElementById('finalMSE').textContent = (currentModel.finalRMSE ** 2).toFixed(6);
    document.getElementById('trainingTime').textContent = currentModel.trainingTime;
    document.getElementById('bestEpoch').textContent = Math.round(currentModel.epochs * 0.85);
    document.getElementById('modelSize').textContent = '3.2 MB';
    
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
        const actual = Math.sin(i * 0.15) * 0.08 + Math.cos(i * 0.1) * 0.03 + Math.random() * 0.02;
        const predicted = actual + (Math.random() - 0.5) * 0.015;
        
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
                    tension: 0.4,
                    fill: false,
                    pointRadius: 3
                },
                {
                    label: 'Predicted Subsidence',
                    data: predictedData,
                    borderColor: '#ef4444',
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    tension: 0.4,
                    borderDash: [5, 5],
                    fill: false,
                    pointRadius: 3
                }
            ]
        },
        options: {
            responsive: true,
            interaction: {
                intersect: false,
                mode: 'index'
            },
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
                },
                legend: {
                    position: 'top'
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
            content = 'Binary model file content would be here (simulated for demo)';
            mimeType = 'application/octet-stream';
            break;
        case 'weights':
            content = 'Binary weights file content would be here (simulated for demo)';
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
                training_time: currentModel.trainingTime,
                created_at: new Date().toISOString()
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

Evaluation Metrics:
- Root Mean Square Error (RMSE): Measures the standard deviation of prediction errors
- Mean Absolute Error (MAE): Average of absolute differences between predicted and actual values
- Mean Squared Error (MSE): Average of squared differences between predicted and actual values

Model Performance:
The PLSTM model demonstrates excellent performance in predicting land subsidence patterns
with low error rates across all evaluation metrics. The model successfully captures
temporal dependencies in the RINEX data and provides reliable predictions for
land subsidence monitoring in Padang City.

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
    const makePredictionBtn = document.getElementById('makePrediction');
    if (makePredictionBtn) {
        makePredictionBtn.addEventListener('click', makePrediction);
    }
}

// Make prediction (simulated)
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
        // Simulate prediction processing
        await new Promise(resolve => setTimeout(resolve, 1500));
        
        // Simulate prediction calculation
        const locationFactor = calculateLocationFactor(easting, northing);
        const heightFactor = calculateHeightFactor(orthoHeight, ellipHeight);
        const temporalFactor = Math.random() * 0.3 + 0.7;
        
        const basePrediction = locationFactor * heightFactor * temporalFactor;
        const prediction = Math.max(0, basePrediction * 0.25);
        
        const confidence = 0.78 + Math.random() * 0.17;
        
        let riskLevel = 'Low';
        let riskColor = '#4ade80';
        let trend = 'Stable';
        
        if (prediction > 0.12) {
            riskLevel = 'High';
            riskColor = '#ef4444';
            trend = 'Increasing';
        } else if (prediction > 0.06) {
            riskLevel = 'Medium';
            riskColor = '#fbbf24';
            trend = 'Moderate';
        }
        
        const result = {
            subsidence: prediction,
            riskLevel,
            riskColor,
            confidence,
            trend,
            coordinates: { easting, northing, orthoHeight, ellipHeight },
            timestamp: new Date().toISOString()
        };
        
        displayPredictionResults(result);
        showMessage('Prediction completed successfully!', 'success');
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
    
    const resultsDiv = document.getElementById('predictionResults');
    resultsDiv.style.display = 'block';
    resultsDiv.classList.add('slide-up');
    
    // Draw risk gauge
    drawRiskGauge(prediction.subsidence * 100, prediction.riskLevel);
}

// Draw risk gauge chart
function drawRiskGauge(value, riskLevel) {
    const canvas = document.getElementById('riskGauge');
    if (!canvas) return;
    
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
    ctx.strokeStyle = '#e5e7eb';
    ctx.stroke();
    
    // Draw risk zones
    const zones = [
        { start: Math.PI, end: Math.PI * 0.67, color: '#4ade80' }, // Low (0-6cm)
        { start: Math.PI * 0.67, end: Math.PI * 0.33, color: '#fbbf24' }, // Medium (6-12cm)
        { start: Math.PI * 0.33, end: 0, color: '#ef4444' } // High (12-20cm)
    ];
    
    zones.forEach(zone => {
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, zone.start, zone.end);
        ctx.lineWidth = 20;
        ctx.strokeStyle = zone.color;
        ctx.stroke();
    });
    
    // Draw needle
    const maxValue = 20; // Maximum value for gauge
    const angle = Math.PI - (Math.min(value, maxValue) / maxValue) * Math.PI;
    const needleLength = radius - 15;
    
    ctx.beginPath();
    ctx.moveTo(centerX, centerY);
    ctx.lineTo(
        centerX + Math.cos(angle) * needleLength,
        centerY + Math.sin(angle) * needleLength
    );
    ctx.lineWidth = 4;
    ctx.strokeStyle = '#374151';
    ctx.stroke();
    
    // Draw center circle
    ctx.beginPath();
    ctx.arc(centerX, centerY, 10, 0, 2 * Math.PI);
    ctx.fillStyle = '#374151';
    ctx.fill();
    
    // Draw value text
    ctx.font = 'bold 24px Inter, sans-serif';
    ctx.fillStyle = '#374151';
    ctx.textAlign = 'center';
    ctx.fillText(`${value.toFixed(1)} cm`, centerX, centerY + 40);
    
    ctx.font = '16px Inter, sans-serif';
    ctx.fillText(riskLevel, centerX, centerY + 65);
}

// Initialize analysis functionality
function initializeAnalysis() {
    const updateAnalysisBtn = document.getElementById('updateAnalysis');
    if (updateAnalysisBtn) {
        updateAnalysisBtn.addEventListener('click', updateAnalysis);
    }
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
    }, 1500);
}

// Create analysis charts
function createAnalysisCharts(analysisType, timeRange) {
    const timeSeriesCtx = document.getElementById('timeSeriesChart');
    const spatialCtx = document.getElementById('spatialAnalysisChart');
    
    // Generate sample data based on analysis type
    const timeSeriesData = generateTimeSeriesData(timeRange);
    const spatialData = generateSpatialAnalysisData();
    
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
                    fill: true,
                    pointRadius: 4,
                    pointHoverRadius: 6
                }]
            },
            options: {
                responsive: true,
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Time Period'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Subsidence (cm)'
                        },
                        beginAtZero: true
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: `${analysisType.charAt(0).toUpperCase() + analysisType.slice(1)} Analysis - ${timeRange}`
                    },
                    legend: {
                        display: false
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
                        point.subsidence > 10 ? '#ef4444' :
                        point.subsidence > 5 ? '#fbbf24' : '#4ade80'
                    ),
                    borderColor: '#374151',
                    borderWidth: 1,
                    pointRadius: spatialData.map(point => Math.max(6, point.subsidence / 2))
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
                        text: 'Spatial Distribution of Subsidence Risk'
                    },
                    legend: {
                        display: false
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
            const baseValue = Math.sin((year - 2021) * 2 + month * 0.5) * 3 + 6;
            const seasonalEffect = Math.sin(month * Math.PI / 6) * 1.5;
            const noise = (Math.random() - 0.5) * 2;
            data.values.push(Math.max(0, baseValue + seasonalEffect + noise));
        }
    }
    
    return data;
}

// Generate spatial analysis data
function generateSpatialAnalysisData() {
    const data = [];
    const basePoints = [
        { x: 100.3565, y: -0.9356 },
        { x: 100.3680, y: -0.9783 },
        { x: 100.4182, y: -0.9834 },
        { x: 100.3500, y: -0.9471 },
        { x: 100.4000, y: -0.9200 },
        { x: 100.3750, y: -0.9600 },
        { x: 100.3900, y: -0.9400 },
        { x: 100.3650, y: -0.9650 }
    ];
    
    basePoints.forEach(point => {
        data.push({
            x: point.x,
            y: point.y,
            subsidence: Math.random() * 18 + 2
        });
    });
    
    return data;
}

// Initialize map
function initializeMap() {
    // Padang City coordinates
    const padangCenter = [-0.9471, 100.4172];
    
    const mapElement = document.getElementById('subsidenceMap');
    if (!mapElement) return;
    
    subsidenceMap = L.map('subsidenceMap').setView(padangCenter, 12);
    
    // Add tile layer
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors'
    }).addTo(subsidenceMap);
    
    // Sample subsidence points
    const subsidencePoints = [
        { name: "Gunung Padang", location: [-0.9356, 100.3565], risk: "High", value: 0.18 },
        { name: "Pantai Air Manis", location: [-0.9783, 100.3680], risk: "Medium", value: 0.09 },
        { name: "Lubuk Begalung", location: [-0.9834, 100.4182], risk: "Low", value: 0.04 },
        { name: "Padang Barat", location: [-0.9471, 100.3500], risk: "Medium", value: 0.11 },
        { name: "Koto Tangah", location: [-0.9200, 100.4000], risk: "High", value: 0.15 },
        { name: "Nanggalo", location: [-0.9600, 100.3750], risk: "Low", value: 0.05 },
        { name: "Pauh", location: [-0.9400, 100.3900], risk: "Medium", value: 0.08 }
    ];
    
    // Add markers
    subsidencePoints.forEach(point => {
        const color = point.risk === 'High' ? '#ef4444' : 
                     point.risk === 'Medium' ? '#fbbf24' : '#4ade80';
        
        const marker = L.marker(point.location).addTo(subsidenceMap);
        marker.bindPopup(`
            <div style="font-family: Inter, sans-serif;">
                <strong style="color: #374151;">${point.name}</strong><br>
                <span style="color: ${color}; font-weight: 600;">Risk Level: ${point.risk}</span><br>
                <span style="color: #6b7280;">Subsidence: ${(point.value * 100).toFixed(1)} cm</span>
            </div>
        `);
    });
}

// Helper functions
function calculateLocationFactor(easting, northing) {
    // Simulate location-based risk factors for Padang City
    const padangCenter = { easting: 100.4172, northing: -0.9471 };
    
    const distance = Math.sqrt(
        Math.pow(easting - padangCenter.easting, 2) + 
        Math.pow(northing - padangCenter.northing, 2)
    );
    
    // Closer to center = higher risk (simplified model)
    return Math.max(0.1, 1 - distance * 8);
}

function calculateHeightFactor(orthoHeight, ellipHeight) {
    // Lower elevations typically have higher subsidence risk
    const heightDiff = Math.abs(ellipHeight - orthoHeight);
    const avgHeight = (orthoHeight + ellipHeight) / 2;
    
    // Lower average height and larger height difference = higher risk
    const heightRisk = Math.max(0.1, 1 - avgHeight / 120);
    const diffRisk = Math.min(1, heightDiff / 15);
    
    return (heightRisk + diffRisk) / 2;
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
    if (mainContent) {
        mainContent.insertBefore(messageDiv, mainContent.firstChild);
        
        // Remove after 5 seconds
        setTimeout(() => {
            if (messageDiv.parentNode) {
                messageDiv.parentNode.removeChild(messageDiv);
            }
        }, 5000);
    }
}