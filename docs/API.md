# API Documentation

## Overview

The SMS Spam Classifier provides a RESTful API for classifying SMS messages as spam or ham (legitimate). The API is built with Flask and returns JSON responses.

## Base URL

```
http://localhost:5000
```

For production deployment:
```
https://your-app-name.herokuapp.com
```

## Endpoints

### 1. Home Page

**Endpoint**: `GET /`

**Description**: Serves the web interface HTML page with a form for message classification.

**Response**: HTML page

**Example**:
```bash
curl http://localhost:5000/
```

---

### 2. Predict Message

**Endpoint**: `POST /predict`

**Description**: Classifies an SMS message as spam or ham and returns confidence scores.

**Request Headers**:
```
Content-Type: application/json
```

**Request Body**:
```json
{
  "message": "Your SMS message text here"
}
```

**Response** (Success - 200 OK):
```json
{
  "message": "Your SMS message text here",
  "prediction": "spam",
  "confidence": {
    "ham": 0.0563,
    "spam": 0.9437
  },
  "processing_time": 0.0234
}
```

**Response Fields**:
- `message` (string): The input message that was classified
- `prediction` (string): Classification result - either "ham" or "spam"
- `confidence` (object): Probability scores for both classes
  - `ham` (float): Confidence score for ham (0.0 to 1.0)
  - `spam` (float): Confidence score for spam (0.0 to 1.0)
- `processing_time` (float): Time taken to process the request in seconds

**Response** (Bad Request - 400):
```json
{
  "error": "Message field is required and cannot be empty"
}
```

**Response** (Server Error - 500):
```json
{
  "error": "Model not loaded. Please train and save a model first."
}
```

or

```json
{
  "error": "Prediction failed: [error details]"
}
```

---

## Usage Examples

### cURL

**Classify a spam message**:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"message": "Congratulations! You won a free iPhone. Click here now!"}'
```

**Classify a ham message**:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"message": "Hey, are we still meeting for lunch tomorrow?"}'
```

### Python

```python
import requests
import json

url = "http://localhost:5000/predict"
headers = {"Content-Type": "application/json"}

# Spam message
data = {
    "message": "URGENT! You have won $1000. Call now to claim your prize!"
}

response = requests.post(url, headers=headers, data=json.dumps(data))
result = response.json()

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence'][result['prediction']]:.2%}")
print(f"Processing time: {result['processing_time']:.4f}s")
```

### JavaScript (Fetch API)

```javascript
const url = 'http://localhost:5000/predict';
const data = {
  message: 'Win a free vacation! Click here to claim your prize!'
};

fetch(url, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify(data)
})
.then(response => response.json())
.then(result => {
  console.log('Prediction:', result.prediction);
  console.log('Confidence:', result.confidence);
  console.log('Processing time:', result.processing_time);
})
.catch(error => console.error('Error:', error));
```

### JavaScript (Axios)

```javascript
const axios = require('axios');

const url = 'http://localhost:5000/predict';
const data = {
  message: 'Thanks for the meeting notes, very helpful!'
};

axios.post(url, data)
  .then(response => {
    const result = response.data;
    console.log('Prediction:', result.prediction);
    console.log('Confidence:', result.confidence);
  })
  .catch(error => console.error('Error:', error));
```

## Error Handling

The API uses standard HTTP status codes:

- `200 OK`: Request successful
- `400 Bad Request`: Invalid request (missing or empty message)
- `500 Internal Server Error`: Server error (model not loaded or prediction failed)

Always check the HTTP status code and handle errors appropriately in your application.

## Rate Limiting

Currently, there is no rate limiting implemented. For production use, consider implementing rate limiting to prevent abuse.

## Authentication

Currently, the API does not require authentication. For production deployment, consider adding API key authentication or OAuth.

## CORS

Cross-Origin Resource Sharing (CORS) is not configured by default. If you need to access the API from a different domain, you'll need to configure CORS in the Flask application.

## Performance

- Average response time: 20-50ms per request
- Throughput: Depends on server resources
- Model loading: Done once on application startup

## Best Practices

1. **Validate input**: Always validate message content before sending
2. **Handle errors**: Implement proper error handling for all API calls
3. **Timeout**: Set appropriate timeout values for API requests
4. **Retry logic**: Implement retry logic for transient failures
5. **Logging**: Log API requests and responses for debugging

## Limitations

- Maximum message length: No explicit limit, but very long messages may take longer to process
- Supported languages: English only (trained on English SMS dataset)
- Model accuracy: ~96% on test data, may vary on real-world data

## Future Enhancements

- Batch prediction endpoint for multiple messages
- Confidence threshold configuration
- Model version selection
- Detailed prediction explanation
- Message preprocessing options
- Support for additional languages
