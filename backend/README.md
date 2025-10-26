# MLP Pattern Recognition - Backend

Backend Python con FastAPI para servir el modelo MLP entrenado.
Proyecto inicializado con [uv](https://docs.astral.sh/uv/), un gestor de paquetes moderno para Python.

## 🚀 Inicio Rápido

### Instalación de Dependencias

```bash
uv sync
```

### Entrenar un Modelo

```bash
# Entrenar modelo de producción (por defecto: 1000 muestras, accuracy ~100%)
uv run python src/train.py
```

### Iniciar el Servidor API

```bash
# Usando uvicorn con hot-reload
uv run uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

El servidor estará disponible en:

- **API**: http://localhost:8000
- **Documentación interactiva**: http://localhost:8000/docs
- **Especificación OpenAPI**: http://localhost:8000/openapi.json

## 📁 Estructura

```
model/
├── src/
│   ├── mlp.py          # Clase MLP con métodos de entrenamiento y predicción
│   ├── train.py        # Script de entrenamiento con versionado
│   └── api.py          # Backend FastAPI
├── trained_models/      # Modelos entrenados con versionado
│   ├── mlp_v1.0_....pkl
│   └── model_registry.json
├── notebook.ipynb       # Experimentación y desarrollo
└── pyproject.toml
```

## 🎯 Endpoints de la API

### `POST /predict`

Predice la letra (b, d, o f) desde un patrón 10x10.

**Request:**

```json
{
  "pattern": [0, 0, 1, ..., 0]  // 100 enteros (0 o 1)
}
```

**Response:**

```json
{
  "letter": "b",
  "probabilities": { "b": 0.98, "d": 0.01, "f": 0.01 },
  "confidence": 0.98
}
```

### `GET /model/info`

Obtiene información del modelo en producción.

### `GET /models/list`

Lista todos los modelos en el registro.

### `GET /health`

Verifica el estado del servicio y del modelo.

## 🔄 Flujo MLOps

### 1. Entrenamiento

```bash
uv run python src/train.py
```

- Genera dataset sintético con ruido
- Entrena el modelo MLP
- Guarda el modelo con versionado automático
- Actualiza el `model_registry.json`

### 2. Versionado de Modelos

Cada modelo se guarda con timestamp, accuracy e hiperparámetros:

```
mlp_v1.0_20251023_012146_acc1.000.pkl
```

### 3. Model Registry

`model_registry.json` mantiene registro de todos los modelos con metadata completa.

### 4. Serving

La API carga automáticamente el modelo marcado como `is_production: true`.

## 🧪 Testing Rápido

```bash
# Test de predicción
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"pattern": [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0]}'

# Info del modelo
curl "http://localhost:8000/model/info"
```

## 🎓 Prácticas MLOps Implementadas

✅ Versionado de modelos con timestamps y métricas  
✅ Model Registry centralizado  
✅ Separación código/artefactos  
✅ API REST con validación Pydantic  
✅ Metadata tracking completo  
✅ Health checks para monitoreo  
✅ Documentación OpenAPI automática  
✅ CORS configurado para frontend
