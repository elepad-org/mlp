# 🧠 Trabajo Páctico MLP

Implementación de un _Multilayer Perceptron_ para detectar patrones en una matriz 10x10.
Trabajo Práctico de Inteligencia Artificial (UTN FRRe).

- [Consigna](https://frre.cvg.utn.edu.ar/pluginfile.php/202733/mod_resource/content/1/TP2025%20-%20MLP.pdf).
- [Tablero Kanban](https://trello.com/b/KvPLKgKd/tp-inteligencia-artificial).
- [Notebook](https://colab.research.google.com/github/elepad-org/mlp/blob/main/model/notebook.ipynb).
- [Repositorio](https://github.com/elepad-org/mlp).

Integrantes del equipo Lambda:

- Aldo Omar **Andres**.
- Sixto Feliciano **Arrejin**.
- Agustín Nicolás **Bravo Pérez**.
- Tobias Alejandro **Maciel Meister**.
- André Leandro **San Lorenzo**.

## 🧑‍💻 Desarrollo

Estructura del Repositorio:

```yaml
mlp/
├── model/        # Notebooks de experimentación (Python)
├── backend/      # API REST + modelo entrenado (Python + FastAPI)
├── app/          # UI web (TypeScript)
└── README.md
```

El modelo será implementado, entrenado y validado con Python en `model/notebook.ipynb`.
Esa notebook se puede abrir en [Google Colab](https://colab.research.google.com/github/elepad-org/mlp/blob/main/model/notebook.ipynb).
Una vez terminado el desarrollo del modelo y la UI, se redactará un informe del trabajo acorde al formato de LNCS de Springer Verlag.

Como experiencia adicional, se busca experimentar con herramientas MLOps y crear una aplicación para usuarios finales.
El resto de este repositorio se dedica a ese objetivo extra.
Se desarrollará una UI web con React + TypeScript para usar el modelo.
Se creará un backend con Python para servir el modelo como una API REST.

## 🚀 Quick Start

### 1. Backend (Python + FastAPI)

```bash
cd backend

# Instalar dependencias
uv sync

# Entrenar el modelo de producción
uv run python src/train.py

# Iniciar el servidor API
cd src && uv run python api.py
```

El backend estará en: [http://localhost:8000](http://localhost:8000)

- Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### 2. Frontend (React + TypeScript)

```bash
cd app

# Instalar dependencias
npm install

# Iniciar en desarrollo
npm run dev
```

El frontend estará en: [http://localhost:5173](http://localhost:5173)

## 📁 Estructura del Repositorio

```yaml
mlp/
├── app/                    # Frontend React + TypeScript
│   ├── src/
│   │   └── components/
│   │       ├── DrawingGrid.tsx      # Grid interactivo 10x10
│   │       └── MLPPredictor.tsx     # Componente de predicción
│   └── package.json
│
├── backend/                # Backend Python + MLOps
│   ├── src/
│   │   ├── mlp.py         # Clase MLP con training/prediction
│   │   ├── train.py       # Script de entrenamiento versionado
│   │   └── api.py         # Backend FastAPI
│   ├── trained_models/     # Modelos guardados con versiones
│   │   ├── mlp_v1.0_*.pkl
│   │   └── model_registry.json
│   └── pyproject.toml
│
├── model/                  # Notebooks de experimentación
│   ├── notebook.ipynb      # Desarrollo original del modelo
│   ├── analysis.ipynb      # Análisis y grid search
│   └── pyproject.toml
│
└── README.md
```

## 🎯 Arquitectura End-to-End

**Stack MLOps End-to-End:**

- 🧠 **Backend**: Python + FastAPI + MLP custom
- 🎨 **Frontend**: React + TypeScript + Vite
- 📦 **Model Registry**: Versionado de modelos con metadata
- 🚀 **Serving**: API REST con validación Pydantic

```text
┌─────────────────┐
│   Frontend      │  React + TypeScript
│  (Port 5173)    │  - DrawingGrid: Dibujar matriz 10x10
└────────┬────────┘  - MLPPredictor: Llamadas HTTP al backend
         │
         │ HTTP POST /predict
         │ { pattern: [0,1,...] }
         ▼
┌─────────────────┐
│   Backend       │  FastAPI + Python
│  (Port 8000)    │  - Validación con Pydantic
└────────┬────────┘  - CORS para frontend
         │
         │ model.classify(X)
         ▼
┌─────────────────┐
│  MLP Model      │  Multilayer Perceptron
│  (pickle)       │  - 100 → 10 → 5 → 3
└─────────────────┘  - Sigmoid activation
                     - Accuracy: ~100%
```

## 🔄 Flujo de Trabajo MLOps

### 1️⃣ Entrenamiento y Versionado

```bash
cd backend
uv run python src/train.py
```

**Output:**

```text
🚀 TRAINING NEW MLP MODEL
📊 Generating dataset with 1000 samples...
🧠 Initializing MLP...
🏋️  Training model...
📈 Evaluating model... Accuracy: 1.0000
💾 Saving model... mlp_v1.0_20251023_acc1.000.pkl
📋 Registry updated
✅ MODEL TRAINING COMPLETE
```

### 2️⃣ Model Registry

Cada modelo se registra con metadata completa:

```json
{
  "models": [
    {
      "version": "v1.0_20251023_012146_acc1.000",
      "filename": "mlp_v1.0_20251023_012146_acc1.000.pkl",
      "created_at": "2025-10-23T01:21:46",
      "accuracy": 1.0,
      "hyperparameters": {
        "activation_type": "sigmoid",
        "learning_rate": 0.1,
        "momentum": 0.1
      },
      "is_production": true
    }
  ]
}
```

### 3️⃣ Serving con FastAPI

```bash
cd backend/src
uv run python api.py
```

**Endpoints disponibles:**

- `POST /predict` - Predicción desde matriz 10x10
- `GET /model/info` - Info del modelo en producción
- `GET /models/list` - Lista todos los modelos
- `GET /health` - Health check

### 4️⃣ Integración Frontend

El componente `MLPPredictor.tsx` consume la API:

```typescript
const response = await fetch(`${API_URL}/predict`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ pattern: flatPattern }),
});

const result = await response.json();
// { letter: "b", probabilities: {...}, confidence: 0.98 }
```

## 🧪 Testing

### Test del Backend

```bash
# Health check
curl http://localhost:8000/health

# Predicción de patrón 'b'
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"pattern": [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0]}'

# Info del modelo
curl http://localhost:8000/model/info
```

## 🎓 Prácticas MLOps Implementadas

✅ **Model Versioning** - Cada modelo con timestamp y métricas  
✅ **Model Registry** - Registro centralizado con metadata  
✅ **Separation of Concerns** - Código separado de artefactos  
✅ **REST API** - Interfaz estándar con FastAPI  
✅ **Validation** - Schemas con Pydantic  
✅ **Serialization** - Pickle para reproducibilidad  
✅ **Health Checks** - Monitoreo de disponibilidad  
✅ **Documentation** - OpenAPI/Swagger automático  
✅ **CORS** - Configurado para integración frontend  
✅ **Metadata Tracking** - Hiperparámetros, arquitectura, datasets

## 📊 Modelo MLP

### Arquitectura

- **Input Layer**: 100 neuronas (matriz 10x10 aplanada)
- **Hidden Layer 1**: 10 neuronas (sigmoid)
- **Hidden Layer 2**: 5 neuronas (sigmoid)
- **Output Layer**: 3 neuronas (b, d, f)

### Entrenamiento

- Dataset: 1000 muestras con ruido 0-30%
- Train/Val split: 80/20
- Optimizer: Backpropagation con momentum
- Loss: Least Mean Squared Error
- Accuracy: ~100% en validación

## 🔮 Roadmap Futuro

- [ ] Docker containerization
- [ ] CI/CD pipeline con GitHub Actions
- [ ] Métricas adicionales (precision, recall, F1)
- [ ] MLflow para experiment tracking
- [ ] A/B testing entre modelos
- [ ] Drift detection en producción
- [ ] Batch predictions endpoint
- [ ] Deployment en cloud (AWS/GCP)
