import React, { useState, useRef, useCallback } from 'react';
import './DrawingGrid.css';

interface DrawingGridProps {
  onPatternChange?: (pattern: number[][]) => void;
}

const DrawingGrid: React.FC<DrawingGridProps> = ({ onPatternChange }) => {
  // Initialize 10x10 grid with all cells empty (0)
  const [grid, setGrid] = useState<number[][]>(() => 
    Array(10).fill(null).map(() => Array(10).fill(0))
  );
  
  const [isDrawing, setIsDrawing] = useState(false);
  const [drawMode, setDrawMode] = useState<'draw' | 'erase'>('draw');
  const gridRef = useRef<HTMLDivElement>(null);

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