import React, { useState } from 'react';
import './MLPPredictor.css';

interface MLPPredictorProps {
  pattern: number[][];
}

// Simulated MLP prediction function
const simulateMLPPrediction = (pattern: number[][]): { letter: string; confidence: number; probabilities: { b: number; d: number; f: number } } => {
  // Convert 2D pattern to 1D array (same as the Python model expects)
  const flatPattern = pattern.flat();
  
  // Count filled pixels to add some basic logic
  const filledPixels = flatPattern.filter(pixel => pixel === 1).length;
  
  // Simulate different predictions based on simple heuristics (just for demo)
  let probabilities = { b: 0.33, d: 0.33, f: 0.34 };
  
  if (filledPixels === 0) {
    // Empty pattern
    probabilities = { b: 0.33, d: 0.33, f: 0.34 };
  } else if (filledPixels < 15) {
    // Few pixels - more likely to be 'f'
    probabilities = { b: 0.2, d: 0.25, f: 0.55 };
  } else if (filledPixels > 25) {
    // Many pixels - more likely to be 'b'
    probabilities = { b: 0.6, d: 0.25, f: 0.15 };
  } else {
    // Medium pixels - more likely to be 'd'
    probabilities = { b: 0.25, d: 0.55, f: 0.2 };
  }
  
  // Add some randomness to make it more realistic
  const noise = (Math.random() - 0.5) * 0.3;
  probabilities.b = Math.max(0.1, Math.min(0.9, probabilities.b + noise));
  probabilities.d = Math.max(0.1, Math.min(0.9, probabilities.d + noise * 0.8));
  probabilities.f = Math.max(0.1, Math.min(0.9, probabilities.f + noise * 0.6));
  
  // Normalize probabilities
  const total = probabilities.b + probabilities.d + probabilities.f;
  probabilities.b /= total;
  probabilities.d /= total;
  probabilities.f /= total;
  
  // Find the letter with highest probability
  const maxProb = Math.max(probabilities.b, probabilities.d, probabilities.f);
  let predictedLetter = 'b';
  if (probabilities.d === maxProb) predictedLetter = 'd';
  else if (probabilities.f === maxProb) predictedLetter = 'f';
  
  return {
    letter: predictedLetter,
    confidence: maxProb,
    probabilities
  };
};

const MLPPredictor: React.FC<MLPPredictorProps> = ({ pattern }) => {
  const [prediction, setPrediction] = useState<ReturnType<typeof simulateMLPPrediction> | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handlePredict = async () => {
    setIsLoading(true);
    
    // Simulate network delay
    await new Promise(resolve => setTimeout(resolve, 800));
    
    const result = simulateMLPPrediction(pattern);
    setPrediction(result);
    setIsLoading(false);
  };

  const isPatternEmpty = pattern.every(row => row.every(cell => cell === 0));

  return (
    <div className="predictor-container">
      <button 
        className="predict-btn"
        onClick={handlePredict}
        disabled={isPatternEmpty || isLoading}
      >
        {isLoading ? (
          <div className="loading-spinner">Analizando...</div>
        ) : (
          'Identificar Letra'
        )}
      </button>
      
      {prediction && !isLoading && (
        <div className="prediction-result">
          <div className="predicted-letter">
            <h3>Predicción:</h3>
            <div className="letter-result">
              <span className="letter">{prediction.letter.toUpperCase()}</span>
              <span className="confidence">
                {(prediction.confidence * 100).toFixed(1)}% confianza
              </span>
            </div>
          </div>
          
          <div className="probabilities">
            <h4>Probabilidades:</h4>
            <div className="prob-bars">
              {Object.entries(prediction.probabilities).map(([letter, prob]) => (
                <div key={letter} className="prob-item">
                  <span className="prob-label">{letter.toUpperCase()}</span>
                  <div className="prob-bar-container">
                    <div 
                      className="prob-bar"
                      style={{ width: `${prob * 100}%` }}
                    />
                  </div>
                  <span className="prob-value">{(prob * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default MLPPredictor;