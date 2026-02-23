# Deployment Guide

This guide covers deploying the SMS Spam Classifier to various platforms.

## Table of Contents

1. [Local Development](#local-development)
2. [Heroku Deployment](#heroku-deployment)
3. [AWS Deployment](#aws-deployment)
4. [Docker Deployment](#docker-deployment)
5. [Google Cloud Platform](#google-cloud-platform)
6. [Production Considerations](#production-considerations)

---

## Local Development

### Prerequisites

- Python 3.9+
- pip
- Git

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/DarshilShukla/Email_Classifier_using_Naive_Bayes.git
   cd Email_Classifier_using_Naive_Bayes
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   python setup_nltk.py
   ```

3. Train the model:
   ```bash
   python main.py
   ```

4. Run the application:
   ```bash
   python app.py
   ```

5. Access at: `http://localhost:5000`

---

## Heroku Deployment

### Prerequisites

- Heroku account
- Heroku CLI installed
- Git

### Step-by-Step Deployment

1. **Login to Heroku**:
   ```bash
   heroku login
   ```

2. **Create a new Heroku app**:
   ```bash
   heroku create your-spam-classifier-app
   ```

3. **Verify deployment files**:
   - `Procfile` ✓
   - `runtime.txt` ✓
   - `requirements.txt` ✓

4. **Add buildpack** (if needed):
   ```bash
   heroku buildpacks:add heroku/python
   ```

5. **Deploy**:
   ```bash
   git push heroku main
   ```

6. **Scale dynos**:
   ```bash
   heroku ps:scale web=1
   ```

7. **Open application**:
   ```bash
   heroku open
   ```

### Heroku Configuration

Set environment variables:
```bash
heroku config:set FLASK_ENV=production
heroku config:set MODEL_PATH=models/spam_classifier_model.pkl
```

View logs:
```bash
heroku logs --tail
```

### Troubleshooting Heroku

**Issue: Slug size too large**
```bash
# Check slug size
heroku builds:info

# Use .slugignore to exclude files
echo "tests/" >> .slugignore
echo "*.hypothesis/" >> .slugignore
```

**Issue: Application timeout**
```bash
# Increase timeout (if using paid dyno)
heroku config:set WEB_CONCURRENCY=2
```

---

## AWS Deployment

### Option 1: AWS Elastic Beanstalk

1. **Install EB CLI**:
   ```bash
   pip install awsebcli
   ```

2. **Initialize EB**:
   ```bash
   eb init -p python-3.9 spam-classifier
   ```

3. **Create environment**:
   ```bash
   eb create spam-classifier-env
   ```

4. **Deploy**:
   ```bash
   eb deploy
   ```

5. **Open application**:
   ```bash
   eb open
   ```

### Option 2: AWS EC2

1. **Launch EC2 instance** (Ubuntu 20.04 LTS)

2. **SSH into instance**:
   ```bash
   ssh -i your-key.pem ubuntu@your-ec2-ip
   ```

3. **Install dependencies**:
   ```bash
   sudo apt update
   sudo apt install python3-pip python3-venv nginx
   ```

4. **Clone repository**:
   ```bash
   git clone https://github.com/DarshilShukla/Email_Classifier_using_Naive_Bayes.git
   cd Email_Classifier_using_Naive_Bayes
   ```

5. **Setup virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   python setup_nltk.py
   ```

6. **Train model**:
   ```bash
   python main.py
   ```

7. **Configure Gunicorn**:
   ```bash
   gunicorn --bind 0.0.0.0:8000 app:app
   ```

8. **Setup Nginx** (reverse proxy):
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;

       location / {
           proxy_pass http://127.0.0.1:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

9. **Setup systemd service**:
   ```ini
   [Unit]
   Description=Spam Classifier
   After=network.target

   [Service]
   User=ubuntu
   WorkingDirectory=/home/ubuntu/Email_Classifier_using_Naive_Bayes
   ExecStart=/home/ubuntu/Email_Classifier_using_Naive_Bayes/venv/bin/gunicorn --bind 0.0.0.0:8000 app:app
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```

---

## Docker Deployment

### Create Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Download NLTK data
RUN python setup_nltk.py

# Train model (optional - can be done separately)
# RUN python main.py

# Expose port
EXPOSE 5000

# Run application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "app:app"]
```

### Build and Run

```bash
# Build image
docker build -t spam-classifier .

# Run container
docker run -p 5000:5000 spam-classifier

# Run with volume for models
docker run -p 5000:5000 -v $(pwd)/models:/app/models spam-classifier
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./models:/app/models
    environment:
      - FLASK_ENV=production
    restart: always
```

Run:
```bash
docker-compose up -d
```

---

## Google Cloud Platform

### App Engine Deployment

1. **Create `app.yaml`**:
   ```yaml
   runtime: python39
   entrypoint: gunicorn -b :$PORT app:app

   instance_class: F2

   automatic_scaling:
     min_instances: 1
     max_instances: 10
   ```

2. **Deploy**:
   ```bash
   gcloud app deploy
   ```

3. **View application**:
   ```bash
   gcloud app browse
   ```

---

## Production Considerations

### Security

1. **HTTPS**: Always use HTTPS in production
2. **API Keys**: Implement API key authentication
3. **Rate Limiting**: Prevent abuse with rate limiting
4. **Input Validation**: Sanitize all user inputs
5. **CORS**: Configure CORS appropriately

### Performance

1. **Caching**: Cache model predictions for common messages
2. **Load Balancing**: Use load balancer for multiple instances
3. **CDN**: Use CDN for static assets
4. **Database**: Store predictions for analytics
5. **Monitoring**: Set up application monitoring

### Monitoring

1. **Application Monitoring**:
   - New Relic
   - Datadog
   - AWS CloudWatch

2. **Error Tracking**:
   - Sentry
   - Rollbar

3. **Logging**:
   - Centralized logging (ELK stack)
   - Log rotation

### Scaling

1. **Horizontal Scaling**: Add more instances
2. **Vertical Scaling**: Increase instance resources
3. **Auto-scaling**: Configure based on traffic
4. **Load Testing**: Test with tools like Apache JMeter

### Backup and Recovery

1. **Model Versioning**: Keep multiple model versions
2. **Database Backups**: Regular backups if using database
3. **Disaster Recovery**: Have rollback plan
4. **Health Checks**: Implement health check endpoints

### Cost Optimization

1. **Right-sizing**: Choose appropriate instance sizes
2. **Auto-scaling**: Scale down during low traffic
3. **Spot Instances**: Use spot instances for non-critical workloads
4. **Caching**: Reduce redundant computations

---

## Environment Variables

Recommended environment variables for production:

```bash
FLASK_ENV=production
MODEL_PATH=models/spam_classifier_model.pkl
VECTORIZER_PATH=models/tfidf_vectorizer.pkl
LOG_LEVEL=INFO
MAX_CONTENT_LENGTH=1048576  # 1MB
```

---

## Health Check Endpoint

Add to `app.py`:

```python
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'timestamp': time.time()
    }), 200
```

---

## Continuous Deployment

### GitHub Actions

See `.github/workflows/tests.yml` for CI/CD pipeline.

Add deployment step:

```yaml
- name: Deploy to Heroku
  if: github.ref == 'refs/heads/main'
  run: |
    git push https://heroku:${{ secrets.HEROKU_API_KEY }}@git.heroku.com/${{ secrets.HEROKU_APP_NAME }}.git main
```

---

## Support

For deployment issues, please open an issue on GitHub.
