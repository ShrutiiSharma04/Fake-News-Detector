# Fake News Detection Backend API

Backend API for fake news detection using BERT transformer model.

## Setup

1. Install Python 3.8+
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the server:
```bash
python app.py
```

The API will be available at `http://localhost:5000`

## API Endpoints

### POST /api/analyze
Analyze text for fake news detection.

**Request:**
```json
{
  "text": "Your news article text here"
}
```

**Response:**
```json
{
  "prediction": "Likely Fake",
  "confidence": 92.5,
  "features": {...},
  "analysis": {...},
  "breakdown": {...}
}
```

### GET /api/health
Check API health status.

## Model

- Base Model: BERT (bert-base-uncased)
- Framework: PyTorch + Hugging Face Transformers
- Approach: Hybrid (BERT + Rule-based)