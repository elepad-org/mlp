import React, { useState, useRef, useCallback } from 'react';
import './DrawingGrid.css';

interface DrawingGridProps {
  onPatternChange?: (pattern: number[][]) => void;
}

// Predefined letter patterns (10x10)
const LETTER_PATTERNS = {
  b: [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  ],
  d: [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  ],
  f: [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  ],
};

type LetterType = keyof typeof LETTER_PATTERNS;

const DrawingGrid: React.FC<DrawingGridProps> = ({ onPatternChange }) => {
  // Initialize 10x10 grid with all cells empty (0)
  const [grid, setGrid] = useState<number[][]>(() => 
    Array(10).fill(null).map(() => Array(10).fill(0))
  );
  
  const [isDrawing, setIsDrawing] = useState(false);
  const [drawMode, setDrawMode] = useState<'draw' | 'erase'>('draw');
  const [selectedLetter, setSelectedLetter] = useState<keyof typeof LETTER_PATTERNS>('b');
  const [distortion, setDistortion] = useState(0);
  const gridRef = useRef<HTMLDivElement>(null);

  // Apply distortion to a pattern
  const applyDistortion = (pattern: number[][], distortionPercent: number): number[][] => {
    const distortedPattern = pattern.map(row => [...row]);
    const totalCells = 100;
    const cellsToFlip = Math.floor((distortionPercent / 100) * totalCells / 2);
    
    for (let i = 0; i < cellsToFlip; i++) {
      const row = Math.floor(Math.random() * 10);
      const col = Math.floor(Math.random() * 10);
      distortedPattern[row][col] = distortedPattern[row][col] === 1 ? 0 : 1;
    }
    
    return distortedPattern;
  };

  const generatePattern = () => {
    const basePattern = LETTER_PATTERNS[selectedLetter];
    const distortedPattern = distortion > 0 
      ? applyDistortion(basePattern, distortion)
      : basePattern.map(row => [...row]);
    
    setGrid(distortedPattern);
    onPatternChange?.(distortedPattern);
  };

  const updateGrid = useCallback((row: number, col: number, value: number) => {
    setGrid(prevGrid => {
      const newGrid = prevGrid.map(r => [...r]);
      newGrid[row][col] = value;
      onPatternChange?.(newGrid);
      return newGrid;
    });
  }, [onPatternChange]);

  const handleMouseDown = (row: number, col: number) => {
    setIsDrawing(true);
    const value = drawMode === 'draw' ? 1 : 0;
    updateGrid(row, col, value);
  };

  const handleMouseEnter = (row: number, col: number) => {
    if (isDrawing) {
      const value = drawMode === 'draw' ? 1 : 0;
      updateGrid(row, col, value);
    }
  };

  const handleMouseUp = () => {
    setIsDrawing(false);
  };

  const clearGrid = () => {
    const emptyGrid = Array(10).fill(null).map(() => Array(10).fill(0));
    setGrid(emptyGrid);
    onPatternChange?.(emptyGrid);
  };

  const toggleDrawMode = () => {
    setDrawMode(prev => prev === 'draw' ? 'erase' : 'draw');
  };

  return (
    <div className="drawing-container">
      {/* Pattern Generator Section */}
      <div className="pattern-generator">
        <h3>Generador de Patrones</h3>
        <div className="generator-controls">
          <div className="letter-selector">
            <label>Letra base:</label>
            <div className="letter-buttons">
              {(['b', 'd', 'f'] as const).map((letter) => (
                <button
                  key={letter}
                  className={`letter-btn ${selectedLetter === letter ? 'selected' : ''}`}
                  onClick={() => setSelectedLetter(letter)}
                >
                  {letter.toUpperCase()}
                </button>
              ))}
            </div>
          </div>
          
          <div className="distortion-slider">
            <label>Distorsión: {distortion}%</label>
            <input
              type="range"
              min="0"
              max="30"
              value={distortion}
              onChange={(e) => setDistortion(Number(e.target.value))}
              className="slider"
            />
          </div>
          
          <button className="generate-btn" onClick={generatePattern}>
            Generar Patrón
          </button>
        </div>
      </div>

      {/* Manual Drawing Controls */}
      <div className="controls">
        <button 
          className={`mode-btn ${drawMode === 'draw' ? 'active' : ''}`}
          onClick={toggleDrawMode}
        >
          {drawMode === 'draw' ? 'Dibujar' : 'Borrar'}
        </button>
        <button className="clear-btn" onClick={clearGrid}>
          Limpiar
        </button>
      </div>
      
      <div 
        className="grid-container"
        ref={gridRef}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        {grid.map((row, rowIndex) => (
          <div key={rowIndex} className="grid-row">
            {row.map((cell, colIndex) => (
              <div
                key={`${rowIndex}-${colIndex}`}
                className={`grid-cell ${cell === 1 ? 'filled' : 'empty'}`}
                onMouseDown={() => handleMouseDown(rowIndex, colIndex)}
                onMouseEnter={() => handleMouseEnter(rowIndex, colIndex)}
                style={{
                  userSelect: 'none',
                  WebkitUserSelect: 'none',
                }}
              />
            ))}
          </div>
        ))}
      </div>
      
      <div className="grid-info">
        <p>Dibuja una letra: <strong>b</strong>, <strong>d</strong>, o <strong>f</strong></p>
        <p className="tip">Mantén presionado y arrastra para dibujar</p>
      </div>
    </div>
  );
};

export default DrawingGrid;