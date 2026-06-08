# Django based  Machine Learning Prediction System

A Django-based web application that integrates Machine Learning models for predictive analytics. The system provides prediction services in two domains:

- 🍌 Banana Leaf Disease Prediction
- 🌍 Air Quality Classification

The application includes user authentication, machine learning model integration, prediction history storage, and an interactive web interface.

---

## Features

### User Management
- User Registration
- User Login & Logout
- Session-based Authentication
- Protected Prediction Pages

### Banana Leaf Disease Prediction
Predict diseases affecting banana plants using multiple machine learning algorithms.

Supported disease classes:
- Healthy
- Black Sigatoka
- Yellow Sigatoka
- Panama Disease
- Moko Disease

Input Features:
- Leaf Length
- Leaf Width
- Colour Intensity
- Spots Present
- Moisture Level
- Texture
- Humidity
- Temperature
- Soil Type

### Air Quality Classification

Classifies air quality as:
- Industrial
- Residential

Input Features:
- CO
- NO₂
- SO₂
- O₃
- PM2.5
- PM10

### Machine Learning Models

The system evaluates predictions using:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Naive Bayes

The best-performing model is automatically selected based on stored accuracy scores.

---

## Technology Stack

### Backend
- Python
- Django 4.2.25
- SQLite3

### Machine Learning
- Scikit-learn
- Pandas
- NumPy
- Joblib

### Frontend
- HTML5
- CSS3
- JavaScript

### Version Control
- Git & GitHub

---

## Project Structure

```text
GUI/
│
├── manage.py
├── db.sqlite3
├── retrain_models.py
│
├── GUI/
│   ├── settings.py
│   ├── urls.py
│   ├── asgi.py
│   └── wsgi.py
│
├── basics/
│   ├── models.py
│   ├── views.py
│   ├── urls.py
│   └── templates/
│
├── airquality/
│   ├── views.py
│   ├── urls.py
│   └── templates/
│
├── ml/
│   ├── train_models.py
│   ├── train_airquality.py
│   ├── banana_leaf_dataset.csv
│   ├── airquality_dataset.csv
│   ├── saved_models/
│   └── saved_models_aq/
│
└── ml_models/
```

---

## Database Model

### PredictionRecord

Stores user prediction history.

```python
class PredictionRecord(models.Model):
    user
    leaf_length
    leaf_width
    colour_intensity
    spots_present
    moisture_level
    texture
    humidity
    temperature
    soil_type
    predicted_label
    best_algorithm
    created_at
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/project-name.git
cd project-name
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

or

```bash
pip install django==4.2.25
pip install scikit-learn pandas numpy joblib
```

### Apply Migrations

```bash
python manage.py migrate
```

### Create Admin User

```bash
python manage.py createsuperuser
```

### Run Server

```bash
python manage.py runserver
```

Application will be available at:

```text
http://127.0.0.1:8000/
```

---

## Training Machine Learning Models

### Banana Leaf Models

```bash
python ml/train_models.py
```

### Air Quality Models

```bash
python ml/train_airquality.py
```

### Retrain All Models

```bash
python retrain_models.py
```

---

## Application Workflow

### Banana Leaf Disease Prediction

1. User enters leaf information.
2. Categorical values are encoded.
3. All trained models generate predictions.
4. Model accuracies are compared.
5. Best-performing model is selected.
6. Prediction is displayed.
7. Prediction history is stored.

### Air Quality Prediction

1. User enters pollutant values.
2. All models evaluate the input.
3. Predictions are generated.
4. Best model is selected.
5. Final classification is displayed.

---

## Screenshots

### Authentication
- Login Page
- Signup Page

### Dashboard
- Prediction Selection Interface

### Banana Leaf Prediction
- Input Form
- Prediction Results
- Model Comparison Table

### Air Quality Prediction
- Pollutant Input Form
- Classification Results

---

## Future Enhancements

- Prediction History Dashboard
- Export Results to PDF/CSV
- Real-time Model Retraining
- Feature Importance Visualization
- Model Versioning
- REST API Integration
- Docker Deployment
- PostgreSQL Support

---

## Educational Outcomes

This project demonstrates:

- Django Web Development
- Machine Learning Integration
- User Authentication
- Database Management
- Model Persistence
- Data Preprocessing
- Full Stack Development

---

## License

This project is developed for educational and learning purposes.

---

## Author

**Karthik Baliga**

Artificial Intelligence & Machine Learning Engineering
