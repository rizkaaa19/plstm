// Global variables
let processedData = null;
let subsidenceMap = null;
let currentModel = null;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeNavigation();
    initializeFileUpload();
    initializePrediction();
    initializeMap();
    initializeCharts();
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

// File upload functionality
function initializeFileUpload() {
    const fileInput = document.getElementById('csvFiles');
    const processButton = document.getElementById('processData');
    
    fileInput.addEventListener('change', function() {
        if (this.files.length > 0) {
            processButton.disabled = false;
            processButton.innerHTML = `<i class="fas fa-play"></i> Process ${this.files.length} file(s)`;
        } else {
            processButton.disabled = true;
            processButton.innerHTML = '<i class="fas fa-play"></i> Load and Process Data';
        }
    });
    
    processButton.addEventListener('click', processDataFiles);
}

// Process uploaded data files
async function processDataFiles() {
    const fileInput = document.getElementById('csvFiles');
    const files = fileInput.files;
    
    if (files.length === 0) {
        showMessage('Please select CSV files first.', 'error');
        return;
    }
    
    showLoading(true);
    
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
            updateDataOverview(result.summary);
            showMessage('Data processed successfully!', 'success');
            
            // Enable training
            document.getElementById('startTraining').disabled = false;
        } else {
            showMessage(`Error processing data: ${result.error}`, 'error');
        }
    } catch (error) {
        showMessage(`Error: ${error.message}`, 'error');
    } finally {
        showLoading(false);
    }
}

// Update data overview display
function updateDataOverview(summary) {
    document.getElementById('totalRecords').textContent = summary.totalRecords.toLocaleString();
    document.getElementById('dateRange').textContent = summary.dateRange ? 
        `${summary.dateRange.start} - ${summary.dateRange.end}` : 'N/A';
    document.getElementById('featuresCount').textContent = summary.features;
    document.getElementById('dataStatus').textContent = 'Ready';
    
    document.getElementById('dataOverview').style.display = 'block';
    document.getElementById('dataOverview').classList.add('slide-up');
}

// Configuration value updates
function updateConfigValues() {
    const epochs = document.getElementById('epochs');
    const epochsValue = document.getElementById('epochsValue');
    const sequenceLength = document.getElementById('sequenceLength');
    const sequenceLengthValue = document.getElementById('sequenceLengthValue');
    const validationSplit = document.getElementById('validationSplit');
    const validationSplitValue = document.getElementById('validationSplitValue');
    
    epochs.addEventListener('input', () => {
        epochsValue.textContent = epochs.value;
    });
    
    sequenceLength.addEventListener('input', () => {
        sequenceLengthValue.textContent = sequenceLength.value;
    });
    
    validationSplit.addEventListener('input', () => {
        validationSplitValue.textContent = validationSplit.value;
    });
    
    // Training button
    document.getElementById('startTraining').addEventListener('click', startModelTraining);
}

// Start model training (simulated)
async function startModelTraining() {
    if (!processedData) {
        showMessage('Please process data first.', 'error');
        return;
    }
    
    const epochs = parseInt(document.getElementById('epochs').value);
    const batchSize = parseInt(document.getElementById('batchSize').value);
    const sequenceLength = parseInt(document.getElementById('sequenceLength').value);
    const validationSplit = parseFloat(document.getElementById('validationSplit').value);
    
    showLoading(true);
    document.getElementById('trainingProgress').style.display = 'block';
    
    // Simulate training progress
    const progressFill = document.getElementById('progressFill');
    const trainingMetrics = document.getElementById('trainingMetrics');
    
    for (let epoch = 1; epoch <= epochs; epoch++) {
        await new Promise(resolve => setTimeout(resolve, 100)); // Simulate training time
        
        const progress = (epoch / epochs) * 100;
        progressFill.style.width = `${progress}%`;
        
        // Simulate metrics
        const loss = (0.1 * Math.exp(-epoch / 20) + Math.random() * 0.01).toFixed(4);
        const mae = (0.05 * Math.exp(-epoch / 15) + Math.random() * 0.005).toFixed(4);
        const valLoss = (0.12 * Math.exp(-epoch / 18) + Math.random() * 0.01).toFixed(4);
        const valMae = (0.06 * Math.exp(-epoch / 16) + Math.random() * 0.005).toFixed(4);
        
        trainingMetrics.innerHTML = `
            <strong>Epoch ${epoch}/${epochs}</strong><br>
            Loss: ${loss} | MAE: ${mae}<br>
            Val Loss: ${valLoss} | Val MAE: ${valMae}
        `;
    }
    
    currentModel = {
        trained: true,
        epochs: epochs,
        batchSize: batchSize,
        sequenceLength: sequenceLength,
        validationSplit: validationSplit
    };
    
    showLoading(false);
    showMessage('Model training completed successfully!', 'success');
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
    
    showLoading(true);
    
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
    const angle = Math.PI - (value / 20) * Math.PI; // Scale 0-20cm to 0-π
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
        
        // Custom marker styling would go here
    });
}

// Initialize charts
function initializeCharts() {
    // This would initialize Chart.js charts for analysis
    // For now, we'll create placeholder charts when the analysis page is visited
}

// Utility functions
function showLoading(show) {
    const overlay = document.getElementById('loadingOverlay');
    overlay.style.display = show ? 'flex' : 'none';
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

// Sample data for charts (would be replaced with real data)
function generateSampleTimeSeriesData() {
    const data = [];
    const startDate = new Date('2021-01-01');
    
    for (let i = 0; i < 365 * 4; i += 30) { // Monthly data for 4 years
        const date = new Date(startDate.getTime() + i * 24 * 60 * 60 * 1000);
        const subsidence = Math.sin(i / 100) * 0.05 + Math.random() * 0.02 + 0.03;
        
        data.push({
            x: date.toISOString().split('T')[0],
            y: subsidence * 100 // Convert to cm
        });
    }
    
    return data;
}

// Initialize analysis charts when analysis page is shown
document.addEventListener('DOMContentLoaded', function() {
    const analysisNavBtn = document.querySelector('[data-page="analysis"]');
    
    analysisNavBtn.addEventListener('click', function() {
        setTimeout(() => {
            initializeAnalysisCharts();
        }, 100);
    });
});

function initializeAnalysisCharts() {
    // Time series chart
    const timeSeriesCtx = document.getElementById('timeSeriesChart');
    if (timeSeriesCtx && !timeSeriesCtx.chart) {
        const timeSeriesData = generateSampleTimeSeriesData();
        
        timeSeriesCtx.chart = new Chart(timeSeriesCtx, {
            type: 'line',
            data: {
                datasets: [{
                    label: 'Land Subsidence (cm)',
                    data: timeSeriesData,
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
                        type: 'time',
                        time: {
                            unit: 'month'
                        },
                        title: {
                            display: true,
                            text: 'Date'
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
                        text: 'Land Subsidence Over Time'
                    }
                }
            }
        });
    }
    
    // Spatial distribution chart
    const spatialCtx = document.getElementById('spatialChart');
    if (spatialCtx && !spatialCtx.chart) {
        const spatialData = [
            { x: 100.3565, y: -0.9356, subsidence: 15 },
            { x: 100.3680, y: -0.9783, subsidence: 8 },
            { x: 100.4182, y: -0.9834, subsidence: 3 },
            { x: 100.3500, y: -0.9471, subsidence: 9 },
            { x: 100.4000, y: -0.9200, subsidence: 12 }
        ];
        
        spatialCtx.chart = new Chart(spatialCtx, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Measurement Points',
                    data: spatialData.map(point => ({
                        x: point.x,
                        y: point.y,
                        subsidence: point.subsidence
                    })),
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
                                return `Subsidence: ${point.subsidence} cm`;
                            }
                        }
                    }
                }
            }
        });
    }
}