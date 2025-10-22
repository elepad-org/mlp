import { useState } from "react";
import DrawingGrid from "./components/DrawingGrid";
import MLPPredictor from "./components/MLPPredictor";
import "./App.css";

function App() {
  const [currentPattern, setCurrentPattern] = useState<number[][]>(
    Array(10).fill(null).map(() => Array(10).fill(0))
  );

  const handlePatternChange = (pattern: number[][]) => {
    setCurrentPattern(pattern);
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>MLP Letter Recognition</h1>
        <p className="subtitle">
          Dibuja una letra (<strong>b</strong>, <strong>d</strong>, o <strong>f</strong>) 
          en la matriz 10×10 y deja que el modelo la identifique
        </p>
      </header>
      
      <main className="app-main">
        <div className="horizontal-layout">
          <section className="drawing-section">
            <DrawingGrid onPatternChange={handlePatternChange} />
          </section>
          
          <section className="prediction-section">
            <h2>Predicción</h2>
            <MLPPredictor pattern={currentPattern} />
          </section>
        </div>
      </main>
      
      <footer className="app-footer">
        <p>
          <strong>Trabajo Práctico MLP</strong> - Inteligencia Artificial (UTN FRRe)
        </p>
        <p className="team">Equipo Lambda</p>
      </footer>
    </div>
  );
}

export default App;
