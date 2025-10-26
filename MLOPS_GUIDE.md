# 🎓 MLOps Implementation Guide

Este documento explica la implementación MLOps profesional del proyecto MLP.

## 📋 Tabla de Contenidos

1. [Arquitectura General](#arquitectura-general)
2. [Pipeline de Entrenamiento](#pipeline-de-entrenamiento)
3. [Model Registry](#model-registry)
4. [API de Serving](#api-de-serving)
5. [Integración Frontend](#integración-frontend)
6. [Mejores Prácticas](#mejores-prácticas)

---

## Arquitectura General

```
┌──────────────────────────────────────────────────────────┐
│                    DESARROLLO                            │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  notebook.ipynb ──► Experimentación                      │
│       │                                                  │
│       └──► src/mlp.py ──► Clase productionizada        │
│                                                          │
└──────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│                   ENTRENAMIENTO                          │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  src/train.py ──► Genera datasets                        │
│       │           Entrena MLP                            │
│       │           Evalúa métricas                        │
│       │                                                  │
│       └──► trained_models/                               │
│               ├── mlp_v1.0_*.pkl                         │
│               └── model_registry.json                    │
│                                                          │
└──────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│                     SERVING                              │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  src/api.py ──► FastAPI                                  │
│       │         Carga modelo producción                  │
│       │         Validación Pydantic                      │
│       │         CORS configurado                         │
│       │                                                  │
│       └──► Endpoints:                                    │
│             POST /predict                                │
│             GET  /model/info                             │
│             GET  /health                                 │
│                                                          │
└──────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│                    FRONTEND                              │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  MLPPredictor.tsx ──► Fetch HTTP                         │
│       │               Manejo de errores                  │
│       │               Display resultados                 │
│       │                                                  │
│       └──► Usuario final                                 │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## Pipeline de Entrenamiento

### 1. Generación de Dataset

```python
def generate_dataset(n_samples: int) -> pd.DataFrame:
    """
    - 10% sin ruido
    - 90% con ruido entre 1-30%
    - Distribución uniforme de clases (b, d, f)
    """
```

**Por qué**: Datos sintéticos permiten control total sobre ruido y distribución.

### 2. Split Train/Val

```python
def split_dataset(df: pd.DataFrame, validation_ratio: float):
    """
    - Shuffle con seed fijo (reproducibilidad)
    - Split estratificado
    """
```

**Por qué**: Validación independiente previene overfitting.

### 3. Entrenamiento

```python
mlp = MLP(
    activation_type="sigmoid",
    learning_rate=0.1,
    momentum=0.1,
    seed=42  # Reproducibilidad
)

history = mlp.train(train_data, val_data, tolerance=1e-6)
```

**Por qué**:

- Seed fijo → reproducibilidad
- Tolerance-based early stopping → evita overfitting
- Momentum → convergencia suave

### 4. Evaluación

```python
eval_results = mlp.evaluate(val_data)
# {'accuracy': 1.0, 'correct': 200, 'total': 200}
```

**Por qué**: Métricas en validación miden generalización real.

### 5. Serialización con Metadata

```python
model_data = {
    "params": self.params,           # Pesos y biases
    "hyperparameters": {...},        # Config del entrenamiento
    "metadata": {...},               # Timestamp, arquitectura
    "history": {...}                 # Losses por época
}

with open(filepath, 'wb') as f:
    pickle.dump(model_data, f)
```

**Por qué**:

- Pickle preserva estado completo
- Metadata permite auditoría
- History ayuda en debugging

---

## Model Registry

### Estructura del Registry

```json
{
  "models": [
    {
      "version": "v1.0_20251023_012146_acc1.000",
      "filename": "mlp_v1.0_20251023_012146_acc1.000.pkl",
      "created_at": "2025-10-23T01:21:46.123456",
      "accuracy": 1.0,
      "hyperparameters": {
        "activation_type": "sigmoid",
        "learning_rate": 0.1,
        "momentum": 0.1,
        "tolerance": 1e-6
      },
      "dataset_info": {
        "n_samples": 1000,
        "train_samples": 800,
        "val_samples": 200,
        "validation_ratio": 0.2
      },
      "is_production": true
    }
  ]
}
```

### Versionado Semántico

**Formato**: `mlp_v{MAJOR}.{MINOR}_{TIMESTAMP}_acc{ACCURACY}.pkl`

**Ejemplo**: `mlp_v1.0_20251023_012146_acc1.000.pkl`

- `v1.0`: Versión del algoritmo
- `20251023_012146`: Timestamp UTC
- `acc1.000`: Accuracy redondeado

**Por qué**:

- Timestamp → orden cronológico
- Accuracy en nombre → quick comparison
- Extensión `.pkl` → formato claro

### Funciones del Registry

```python
def register_model(...):
    """Agrega entrada al registry con validación"""

def get_production_model() -> MLP:
    """Carga el modelo marcado como producción"""

def load_model_registry() -> dict:
    """Lee registry desde disco con manejo de errores"""
```

**Por qué**:

- Centralización → single source of truth
- Flag production → deployment claro
- Funciones helper → abstracción

---

## API de Serving

### Endpoints Diseñados

#### POST /predict

```python
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Input: {"pattern": [0,1,0,...]}  # 100 ints
    Output: {
        "letter": "b",
        "probabilities": {"b": 0.98, "d": 0.01, "f": 0.01},
        "confidence": 0.98
    }
    """
```

**Características**:

- ✅ Validación automática con Pydantic
- ✅ Type hints para autocomplete
- ✅ Error handling con HTTPException
- ✅ Async para concurrencia

#### GET /model/info

```python
@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """
    Devuelve metadata del modelo en producción:
    - Version, accuracy, timestamp
    - Hiperparámetros
    - Arquitectura
    """
```

**Por qué**: Transparencia sobre qué modelo está sirviendo.

#### GET /health

```python
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check para monitoreo:
    - Status: healthy/degraded
    - Model loaded: true/false
    """
```

**Por qué**: Kubernetes/Docker health probes.

### Lazy Loading del Modelo

```python
_model: MLP | None = None

def get_model() -> MLP:
    global _model
    if _model is None:
        _model = get_production_model()
    return _model
```

**Por qué**:

- No carga modelo hasta primera petición
- Singleton → una sola carga en memoria
- Startup más rápido

### CORS Configuration

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En prod: específico
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Por qué**: Frontend en localhost:5173 puede llamar a backend en localhost:8000.

---

## Integración Frontend

### Flow de Predicción

```typescript
// 1. Usuario dibuja en grid 10x10
const pattern: number[][] = [...]; // 2D array

// 2. Aplanar a 1D
const flatPattern = pattern.flat(); // [0,1,0,...]

// 3. HTTP POST
const response = await fetch(`${API_URL}/predict`, {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({ pattern: flatPattern })
});

// 4. Parse response
const result = await response.json();
// { letter: "b", probabilities: {...}, confidence: 0.98 }

// 5. Display en UI
setPrediction(result);
```

### Manejo de Errores

```typescript
try {
  const result = await predictWithMLP(pattern);
  setPrediction(result);
} catch (err) {
  setError(err.message);
  // Display: "Backend no disponible en http://localhost:8000"
}
```

**Por qué**: UX clara cuando backend está caído.

### Environment Variables

```bash
# .env
VITE_API_URL=http://localhost:8000
```

```typescript
const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
```

**Por qué**: Fácil cambio entre dev/staging/prod.

---

## Mejores Prácticas

### ✅ Reproducibilidad

```python
SEED = 42
RNG = np.random.default_rng(SEED)

# En MLP
def __init__(self, ..., seed: int = 42):
    self.rng = np.random.default_rng(seed)
```

**Beneficio**: Mismo dataset → mismo modelo → mismas métricas.

### ✅ Separación de Concerns

```
src/
  mlp.py     → Modelo puro (sin I/O, sin API)
  train.py   → Training logic + registry
  api.py     → Serving logic
```

**Beneficio**: Testing y refactoring más fácil.

### ✅ Type Hints

```python
def train(
    self,
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    tolerance: float = 1e-6
) -> dict:
```

**Beneficio**: IDE autocomplete + static type checking.

### ✅ Validation with Pydantic

```python
class PredictionRequest(BaseModel):
    pattern: List[int] = Field(..., min_length=100, max_length=100)

    @field_validator('pattern')
    def validate_pattern(cls, v):
        if not all(x in [0, 1] for x in v):
            raise ValueError("Only 0s and 1s allowed")
        return v
```

**Beneficio**: API robusta, errores claros al cliente.

### ✅ Graceful Degradation

```python
if _model is None:
    raise HTTPException(
        status_code=503,
        detail="No production model available"
    )
```

**Beneficio**: API responde incluso sin modelo (con error claro).

### ✅ Logging

```python
print("🔄 Loading production model...")
# → En prod: logger.info("Loading production model")
```

**Beneficio**: Observabilidad en producción.

---

## Flujo Completo End-to-End

### Día 1: Setup Inicial

```bash
# 1. Instalar dependencias
./dev.sh install

# 2. Entrenar primer modelo
./dev.sh train

# 3. Verificar registry
cat model/trained_models/model_registry.json
```

### Día 2: Desarrollo

```bash
# Terminal 1: Backend
./dev.sh backend

# Terminal 2: Frontend
./dev.sh frontend

# Terminal 3: Testing
./dev.sh check
./dev.sh test
```

### Día N: Reentrenamiento

```bash
# 1. Modificar hiperparámetros en train.py
# 2. Entrenar nuevo modelo
./dev.sh train

# 3. Automáticamente usa nuevo modelo (is_production=true)
# 4. Verificar mejora
curl http://localhost:8000/model/info
```

---

## Próximos Pasos

### Corto Plazo

- [ ] Unit tests para MLP
- [ ] Integration tests para API
- [ ] Logging estructurado (JSON)

### Mediano Plazo

- [ ] Docker Compose
- [ ] CI/CD con GitHub Actions
- [ ] Métricas por clase (precision, recall)

### Largo Plazo

- [ ] MLflow integration
- [ ] Kubernetes deployment
- [ ] Monitoring con Prometheus/Grafana
- [ ] A/B testing framework

---

## Referencias

- **FastAPI**: https://fastapi.tiangolo.com/
- **Pydantic**: https://docs.pydantic.dev/
- **MLOps Principles**: https://ml-ops.org/
- **Model Versioning**: https://dvc.org/doc/use-cases/versioning-data-and-model-files

---

**Autor**: Equipo Lambda - UTN FRRe  
**Fecha**: Octubre 2025  
**Versión**: 1.0
