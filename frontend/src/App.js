import React, { useState } from 'react';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [filePreview, setFilePreview] = useState(null);
  const [selectedAge, setSelectedAge] = useState('20-29');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  
  const AGE_GROUPS = ["0-5", "6-12", "13-19", "20-29", "30-39", "40-49", "50-69", "70+"];
  
  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setFilePreview(URL.createObjectURL(selectedFile));
      setResult(null);
    }
  };
  
  const handleDrop = (event) => {
    event.preventDefault();
    
    if (event.dataTransfer.files && event.dataTransfer.files[0]) {
      const droppedFile = event.dataTransfer.files[0];
      setFile(droppedFile);
      setFilePreview(URL.createObjectURL(droppedFile));
      setResult(null);
    }
  };
  
  const handleDragOver = (event) => {
    event.preventDefault();
  };
  
  const handleAgeChange = (age) => {
    setSelectedAge(age);
  };
  
// In App.js, update the handleSubmit function:

  const handleSubmit = async () => {
    if (!file) return;
    
    setLoading(true);
    
    const formData = new FormData();
    formData.append('image', file);
    formData.append('targetAge', selectedAge);
    
    try {
      const response = await fetch('http://localhost:5000/api/transform', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      
      const data = await response.json();
      console.log("Response data:", data); // For debugging
      
      setResult({
        originalImage: data.originalImage,
        transformedImage: data.transformedImages[selectedAge],
        targetAge: selectedAge,
        allTransformedImages: data.transformedImages
      });
    } catch (error) {
      console.error('Error:', error);
      alert('Failed to process image. Please try again.');
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <div className="app-container">
      <h1 className="app-title">Age Progression App</h1>
      
      {/* Upload Section */}
      <div className="card">
        <h2 className="card-title">Upload Your Photo</h2>
        
        <div 
          className="upload-area"
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onClick={() => document.getElementById('file-input').click()}
        >
          <input
            id="file-input"
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            style={{ display: 'none' }}
          />
          <p>Drag & drop an image here, or click to select one</p>
          <button className="btn primary-btn">
            Select Image
          </button>
        </div>
        
        {filePreview && (
          <div className="preview-container">
            <p className="preview-title">Selected Image:</p>
            <img 
              src={filePreview} 
              alt="Preview" 
              className="preview-image" 
            />
          </div>
        )}
      </div>
      
      {/* Age Selection */}
      <div className="card">
        <h2 className="card-title">Select Target Age</h2>
        
        <div className="age-buttons">
          {AGE_GROUPS.map((age) => (
            <button
              key={age}
              onClick={() => handleAgeChange(age)}
              className={`age-btn ${selectedAge === age ? 'selected' : ''}`}
            >
              {age}
            </button>
          ))}
        </div>
        
        <div className="submit-container">
          <button
            onClick={handleSubmit}
            disabled={!file || loading}
            className={`btn submit-btn ${(!file || loading) ? 'disabled' : ''}`}
          >
            {loading ? "Processing..." : "Transform Image"}
          </button>
        </div>
      </div>
      
      {/* Results Section */}
      {result && (
        <div className="card">
          <h2 className="card-title">Results</h2>
          <p className="result-info">Age transformation to {result.targetAge} complete!</p>
          
          <div className="results-grid">
            <div className="result-card">
              <div className="result-header">
                <h3>Original Image</h3>
              </div>
              <img
                src={result.originalImage}
                alt="Original"
                className="result-image"
              />
            </div>
            
            <div className="result-card">
              <div className="result-header">
                <h3>Transformed Image</h3>
              </div>
              <img
                src={result.transformedImage}
                alt="Transformed"
                className="result-image"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;