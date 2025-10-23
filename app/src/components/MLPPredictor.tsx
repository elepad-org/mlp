import React, { useState } from 'react';
import './MLPPredictor.css';

interface MLPPredictorProps {
  pattern: number[][];
}

// Simulated MLP prediction function
const simulateMLPPrediction = (pattern: number[][]): { letter: string } => {
  // Convert 2D pattern to 1D array (same as the Python model expects)
  const flatPattern = pattern.flat();
  
  // Count filled pixels to add some basic logic
  const filledPixels = flatPattern.filter(pixel => pixel === 1).length;
  
  // Simple heuristic-based prediction (just for demo)
  let predictedLetter = 'b'; // default
  
  if (filledPixels === 0) {
    // Empty pattern - random choice
    predictedLetter = ['b', 'd', 'f'][Math.floor(Math.random() * 3)];
  } else if (filledPixels < 15) {
    // Few pixels - more likely to be 'f'
    predictedLetter = Math.random() < 0.6 ? 'f' : (Math.random() < 0.5 ? 'b' : 'd');
  } else if (filledPixels > 25) {
    // Many pixels - more likely to be 'b'
    predictedLetter = Math.random() < 0.6 ? 'b' : (Math.random() < 0.5 ? 'd' : 'f');
  } else {
    // Medium pixels - more likely to be 'd'
    predictedLetter = Math.random() < 0.6 ? 'd' : (Math.random() < 0.5 ? 'b' : 'f');
  }
  
  return { letter: predictedLetter };
};

const MLPPredictor: React.FC<MLPPredictorProps> = ({ pattern }) => {
  const [prediction, setPrediction] = useState<ReturnType<typeof simulateMLPPrediction> | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handlePredict = async () => {
    setIsLoading(true);
    
    // Simulate network delay
    await new Promise(resolve => setTimeout(resolve, 400));
    
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
          <div className="predicted-letter-simple">
            <h3>La letra es:</h3>
            <div className="letter-display">
              {prediction.letter} 
            </div> 
          </div>
        </div>
      )}
    </div>
  );
};

export default MLPPredictor;