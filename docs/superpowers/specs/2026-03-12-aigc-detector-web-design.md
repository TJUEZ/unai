# AIGC Text Detector Web Interface - Design Spec

## Project Overview

- **Project Name**: AIGC Text Detector (中文文本AI生成检测器)
- **Type**: Web Application (Flask + HTML)
- **Core Functionality**: Detect whether a given Chinese text is AI-generated or human-written
- **Target Users**: Content creators, educators, researchers who need to verify text authenticity

## Architecture

### Tech Stack
- **Backend**: Flask (Python web framework)
- **Frontend**: Single HTML page with embedded CSS/JavaScript
- **ML Model**: Chinese RoBERTa (chinese-roberta-wwm-ext) with classification head
- **Model Files**: Local files in project root (`pytorch_model.bin`, `config.json`, `tokenizer_config.json`, `vocab.txt`, `special_tokens_map.json`)

### Components

1. **Flask API Server** (`app.py`)
   - `/api/detect` endpoint: POST request with text, returns AIGC probability
   - CORS enabled for local development
   - Model loaded once at startup

2. **Frontend** (`templates/index.html`)
   - Text input textarea
   - Detect button
   - Result display with visual indicator
   - Clean Modern design (light gray #f5f7fa, blue accents #4A90E2)

## UI/UX Specification

### Layout
- Centered container (max-width: 800px)
- Header with title
- Text input area (textarea, 200px height)
- Action button
- Result card (hidden initially, shown after detection)

### Visual Design

**Color Palette**
- Background: #f5f7fa (light gray)
- Container: #ffffff (white)
- Primary: #4A90E2 (blue)
- Primary Hover: #357ABD
- Text Primary: #2c3e50
- Text Secondary: #7f8c8d
- AI Generated: #e74c3c (red) - high confidence
- Human Written: #27ae60 (green) - high confidence

**Typography**
- Font: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif
- Title: 28px, bold
- Body: 16px
- Result percentage: 48px, bold

### Interactions
- Button click → loading state → show result
- Result shows: "AI生成概率" (AIGC Probability) with percentage
- Visual bar showing probability (0-100%)
- Color gradient from green (0%) to red (100%)

## Functionality Specification

### Core Features

1. **Text Input**
   - User can paste or type Chinese text
   - Maximum length: 512 tokens (model limit)

2. **Detection**
   - Click "检测" button to analyze
   - Send text to Flask backend
   - Backend tokenizes and runs inference
   - Returns probability score (0-1)

3. **Result Display**
   - Show "AI生成概率" percentage
   - Visual progress bar
   - Color indication (green = human, red = AI)

### API Specification

**Endpoint**: `POST /api/detect`

Request:
```json
{
  "text": "要检测的文本内容"
}
```

Response:
```json
{
  "success": true,
  "probability": 0.85,
  "result": "AI生成的文本",
  "text_length": 100
}
```

Error Response:
```json
{
  "success": false,
  "error": "错误信息"
}
```

## Model Configuration

- Model type: BertForSequenceClassification
- Base model: chinese-roberta-wwm-ext
- Max sequence length: 512
- Output: Binary classification (AI-generated vs Human-written)
- Probability interpretation: Higher = more likely AI-generated

## Acceptance Criteria

1. ✅ Flask server starts without errors
2. ✅ Frontend loads at http://localhost:5000
3. ✅ User can input Chinese text
4. ✅ Clicking detect sends request to backend
5. ✅ Backend returns probability score
6. ✅ Result displays correctly with visual indicator
7. ✅ Model runs entirely locally (no external API calls)
8. ✅ Works with Chinese text input

## File Structure

```
AIGC_detector_zhv3/
├── app.py                 # Flask application
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html         # Frontend HTML
├── pytorch_model.bin       # Model weights
├── config.json            # Model config
├── tokenizer_config.json  # Tokenizer config
├── vocab.txt              # Vocabulary
├── special_tokens_map.json # Special tokens
└── AIGC_text_detector/    # Original detector code
```
