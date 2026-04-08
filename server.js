

// server.js
const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const csv = require('csv-parser');
const axios = require('axios');

const app = express();
const PORT = 3000;

app.use(cors());
app.use(express.static(__dirname));

const csvFilePath = path.join(__dirname, 'processed/final_processed_data.csv');
let rawData = [];

// Load CSV into memory on startup
fs.createReadStream(csvFilePath)
  .pipe(csv())
  .on('data', (row) => {
    rawData.push({
      disease: row.Disease,
      fever: parseInt(row.Fever),
      cough: parseInt(row.Cough),
      fatigue: parseInt(row.Fatigue),
      difficultyBreathing: parseInt(row['Difficulty Breathing']),
      age: parseInt(row.Age),
      gender: parseInt(row.Gender),
      bloodPressure: parseInt(row['Blood Pressure']),
      cholesterolLevel: parseInt(row['Cholesterol Level']),
      outcome: parseInt(row['Outcome Variable']),
    });
  })
  .on('end', () => {
    console.log('✅ CSV data loaded successfully');
  });

// Add simple predicted outcome based on symptoms
function cleanseData(data) {
  return data.map(record => ({
    ...record,
    predictedOutcome:
      (record.fever && record.cough) || record.fatigue || record.difficultyBreathing ? 1 : 0
  }));
}

// GET /data endpoint — serves cleaned data + metrics + outcome analysis
app.get('/data', async (req, res) => {
  const cleanedData = cleanseData(rawData);

  // Count outcomes: [countOutcome0, countOutcome1]
  const outcomeCounts = cleanedData.reduce(
    (acc, record) => {
      acc[record.outcome] += 1;
      return acc;
    },
    [0, 0]
  );

  // Default quality metrics if Flask backend not available
  let qualityMetrics = {
    random_forest: { accuracy: 'N/A', precision: 'N/A', recall: 'N/A', f1Score: 'N/A' },
    svm: { accuracy: 'N/A', precision: 'N/A', recall: 'N/A', f1Score: 'N/A' },
    ensemble: { accuracy: 'N/A', precision: 'N/A', recall: 'N/A', f1Score: 'N/A' }
  };

  try {
    const flaskResponse = await axios.get('http://localhost:5000/metrics');
    const rf = flaskResponse.data.random_forest;
    const svm = flaskResponse.data.svm;
    const ensemble = flaskResponse.data.ensemble;
    
    if (ensemble) qualityMetrics.ensemble = ensemble;
    if (rf) qualityMetrics.random_forest = rf;
    if (svm) qualityMetrics.svm = svm;
    
  } catch (error) {
    console.error('❌ Error fetching metrics from Flask:', error.message);
  }

  res.json({ cleanedData, outcomeAnalysis: outcomeCounts, qualityMetrics });
});

// GET /metrics endpoint — proxy to Flask backend metrics
app.get('/metrics', async (req, res) => {
  try {
    const response = await axios.get('http://localhost:5000/metrics');
    res.json(response.data);
  } catch (error) {
    console.error('❌ Error fetching metrics:', error.message);
    res.status(500).json({ error: 'Could not fetch metrics from model server.' });
  }
});

// Serve frontend HTML
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'frontend/index.html'));
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Node.js server running at http://localhost:${PORT}`);
});
