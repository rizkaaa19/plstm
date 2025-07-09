const express = require('express');
const cors = require('cors');
const multer = require('multer');
const csv = require('csv-parser');
const fs = require('fs');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// Configure multer for file uploads
const upload = multer({ dest: 'uploads/' });

// Serve static files
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// API endpoint to process CSV data
app.post('/api/process-data', upload.array('csvFiles'), async (req, res) => {
    try {
        const files = req.files;
        let allData = [];

        // Process each uploaded CSV file
        for (const file of files) {
            const data = await processCSVFile(file.path);
            allData = allData.concat(data);
        }

        // Clean up uploaded files
        files.forEach(file => {
            fs.unlinkSync(file.path);
        });

        // Process and clean the data
        const processedData = processLandSubsidenceData(allData);

        res.json({
            success: true,
            data: processedData,
            summary: {
                totalRecords: processedData.length,
                dateRange: getDateRange(processedData),
                features: Object.keys(processedData[0] || {}).length
            }
        });
    } catch (error) {
        console.error('Error processing data:', error);
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

// API endpoint for predictions
app.post('/api/predict', (req, res) => {
    try {
        const { easting, northing, orthoHeight, ellipHeight } = req.body;
        
        // Simplified prediction logic (in real implementation, use trained model)
        const prediction = Math.random() * 0.2; // Random value between 0-0.2m
        const confidence = 0.8 + Math.random() * 0.15; // Random confidence 80-95%
        
        let riskLevel = 'Low';
        let riskColor = '#4CAF50';
        
        if (prediction > 0.1) {
            riskLevel = 'High';
            riskColor = '#F44336';
        } else if (prediction > 0.05) {
            riskLevel = 'Medium';
            riskColor = '#FF9800';
        }

        res.json({
            success: true,
            prediction: {
                subsidence: prediction,
                riskLevel,
                riskColor,
                confidence,
                coordinates: { easting, northing, orthoHeight, ellipHeight }
            }
        });
    } catch (error) {
        console.error('Error making prediction:', error);
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

// Helper functions
function processCSVFile(filePath) {
    return new Promise((resolve, reject) => {
        const results = [];
        fs.createReadStream(filePath)
            .pipe(csv({ separator: ';' }))
            .on('data', (data) => results.push(data))
            .on('end', () => resolve(results))
            .on('error', reject);
    });
}

function processLandSubsidenceData(rawData) {
    return rawData.map(row => {
        // Clean and process each row
        const processed = {};
        
        Object.keys(row).forEach(key => {
            const cleanKey = key.trim().replace(/[^\w\s\[\]._-]/g, '');
            let value = row[key];
            
            // Convert DMS coordinates to decimal degrees
            if (cleanKey.includes('Latitude') || cleanKey.includes('Longitude')) {
                value = dmsToDecimal(value);
            }
            
            // Convert numeric values with thousand separators
            if (cleanKey.includes('[m]')) {
                value = convertToFloat(value);
            }
            
            // Parse dates
            if (cleanKey.includes('Time')) {
                value = new Date(value).toISOString();
            }
            
            processed[cleanKey] = value;
        });
        
        return processed;
    }).filter(row => Object.values(row).some(val => val !== null && val !== undefined));
}

function dmsToDecimal(dmsStr) {
    if (!dmsStr || dmsStr === '') return null;
    
    const match = String(dmsStr).match(/(\d+)° (\d+)'.*?([\d.]+)" (\w)/);
    if (match) {
        const [, degrees, minutes, seconds, direction] = match;
        let dd = parseFloat(degrees) + parseFloat(minutes)/60 + parseFloat(seconds)/3600;
        if (direction === 'S' || direction === 'W') dd *= -1;
        return dd;
    }
    return null;
}

function convertToFloat(value) {
    try {
        return parseFloat(String(value).replace(/\./g, '').replace(',', '.'));
    } catch {
        return null;
    }
}

function getDateRange(data) {
    const dates = data
        .map(row => row['Start Time'])
        .filter(date => date)
        .map(date => new Date(date))
        .sort();
    
    if (dates.length === 0) return null;
    
    return {
        start: dates[0].getFullYear(),
        end: dates[dates.length - 1].getFullYear()
    };
}

app.listen(PORT, () => {
    console.log(`Land Subsidence Detection System running on http://localhost:${PORT}`);
});