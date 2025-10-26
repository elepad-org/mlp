import React, { useState } from "react";
import "./MLPPredictor.css";

interface MLPPredictorProps {
  pattern: number[][];
}

interface PredictionResult {
  letter: string;
  probabilities: {
    b: number;
    d: number;
    f: number;
  };
  confidence: number;
}

// API endpoint (change this to your backend URL)
const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

// Function to call the real MLP backend
const predictWithMLP = async (
  pattern: number[][],
): Promise<PredictionResult> => {
  // Convert 2D pattern to 1D array (same as the Python model expects)
  const flatPattern = pattern.flat();

  const response = await fetch(`${API_URL}/predict`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ pattern: flatPattern }),
  });

  if (!response.ok) {
    throw new Error(`Prediction failed: ${response.statusText}`);
  }

  return await response.json();
};

const MLPPredictor: React.FC<MLPPredictorProps> = ({ pattern }) => {
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handlePredict = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const result = await predictWithMLP(pattern);
      setPrediction(result);
    } catch (err) {
      const errorMessage =
        err instanceof Error ? err.message : "Error desconocido";
      setError(errorMessage);
      console.error("Prediction error:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const isPatternEmpty = pattern.every((row) =>
    row.every((cell) => cell === 0),
  );

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
          "Identificar Letra"
        )}
      </button>

      {error && (
        <div className="error-message">
          <p>❌ Error: {error}</p>
          <p style={{ fontSize: "0.9em", marginTop: "0.5rem" }}>
            Asegúrate de que el backend esté corriendo en {API_URL}
          </p>
        </div>
      )}

      {prediction && !isLoading && !error && (
        <div className="prediction-result">
          <div className="predicted-letter-simple">
            <h3>La letra es:</h3>
            <div className="letter-display">{prediction.letter}</div>
            <div className="confidence-info">
              <p>Confianza: {(prediction.confidence * 100).toFixed(1)}%</p>
              <div className="probabilities">
                <div>b: {(prediction.probabilities.b * 100).toFixed(1)}%</div>
                <div>d: {(prediction.probabilities.d * 100).toFixed(1)}%</div>
                <div>f: {(prediction.probabilities.f * 100).toFixed(1)}%</div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default MLPPredictor;
