# Django Machine Learning Prediction System - Complete Project Documentation

## 📋 Executive Summary

This is a **Django-based web application** that implements machine learning models for **predictive analytics**. The system provides users with the ability to make predictions using trained ML models for two distinct domains:

1. **Banana Leaf Disease Prediction** - Predicts disease types affecting banana plants
2. **Air Quality Classification** - Classifies air quality types based on pollution levels

The application features a full-stack implementation with user authentication, model training pipelines, and an interactive web interface for making predictions.

---

## 🏗️ Project Architecture Overview

```
GUI/ (Project Root)
├── manage.py                          # Django project management script
├── db.sqlite3                         # SQLite database
├── retrain_models.py                  # Script to retrain ML models
│
├── GUI/                               # Main Django configuration
│   ├── __init__.py
│   ├── settings.py                    # Project settings (installed apps, middleware, etc.)
│   ├── urls.py                        # Main URL routing
│   ├── asgi.py                        # ASGI configuration
│   └── wsgi.py                        # WSGI configuration
│
├── basics/                            # Core app for user management & banana leaf prediction
│   ├── __init__.py
│   ├── admin.py                       # Django admin configuration
│   ├── apps.py                        # App configuration
│   ├── models.py                      # Database models
│   ├── views.py                       # View functions for authentication & predictions
│   ├── urls.py                        # App-specific URL routing
│   ├── tests.py                       # Unit tests
│   ├── migrations/                    # Database migrations
│   │   ├── __init__.py
│   │   ├── 0001_initial.py
│   │   ├── 0002_rename_emploee_table_employee_table.py
│   │   └── 0003_predictionrecord_delete_employee_table.py
│   └── templates/                     # HTML templates
│       ├── index.html                 # Home page
│       ├── login.html                 # Login page
│       ├── signup.html                # Registration page
│       ├── dashboard.html             # Main dashboard (prediction selection)
│       └── predict.html               # Banana leaf prediction form & results
│
├── airquality/                        # Air quality prediction app
│   ├── __init__.py
│   ├── views.py                       # Air quality prediction views
│   ├── urls.py                        # Air quality URL routing
│   └── templates/
│       └── airquality/
│           ├── aq_home.html           # Air quality home page
│           └── aq_predict.html        # Air quality prediction form & results
│
├── ml/                                # Machine learning module
│   ├── banana_leaf_dataset.csv        # Dataset for banana leaf disease
│   ├── airquality_dataset.csv         # Dataset for air quality
│   ├── train_models.py                # Script to train banana leaf models
│   ├── train_airquality.py            # Script to train air quality models
│   ├── saved_models/                  # Trained banana leaf models
│   │   ├── logistic_regression.pkl
│   │   ├── decision_tree.pkl
│   │   ├── random_forest.pkl
│   │   ├── svm.pkl
│   │   ├── knn.pkl
│   │   ├── naive_bayes.pkl
│   │   ├── label_encoder.pkl          # Encoder for disease labels
│   │   ├── le_texture.pkl             # Encoder for texture categories
│   │   ├── le_soil.pkl                # Encoder for soil type categories
│   │   ├── le_color.pkl               # Encoder for color intensity
│   │   └── accuracies.pkl             # Stored accuracy scores
│   └── saved_models_aq/               # Trained air quality models
│       ├── lr_aq.pkl
│       ├── dt_aq.pkl
│       ├── rf_aq.pkl
│       ├── svm_aq.pkl
│       ├── knn_aq.pkl
│       ├── nb_aq.pkl
│       ├── label_encoder_aq.pkl       # Encoder for air quality labels
│       └── accuracies_aq.pkl          # Stored accuracy scores
│
└── ml_models/                         # Placeholder for additional models
```

---

## 🔧 Technology Stack

### Backend

- **Framework**: Django 4.2.25
- **Database**: SQLite3
- **Language**: Python 3.x

### Machine Learning

- **scikit-learn**: For ML algorithms (Logistic Regression, Decision Tree, Random Forest, SVM, KNN, Naive Bayes)
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **joblib**: Model serialization and loading

### Frontend

- **HTML5**: Template structure
- **CSS3**: Styling and responsive design
- **JavaScript**: (Minimal for form handling)

### Tools

- **Git/GitHub**: Version control
- **Virtual Environment**: Python venv for dependency isolation

---

## 📊 Database Models

### PredictionRecord Model

Located in `basics/models.py`

```python
class PredictionRecord(models.Model):
    user                = ForeignKey(User)  # Links to Django's User model
    leaf_length         = FloatField()      # Length of banana leaf in cm
    leaf_width          = FloatField()      # Width of banana leaf in cm
    colour_intensity    = CharField()       # Categorical: 'Dark', 'Light', 'Medium'
    spots_present       = IntegerField()    # Binary: 0 (no) or 1 (yes)
    moisture_level      = FloatField()      # Moisture % of leaf
    texture             = CharField()       # Categorical: 'Rough', 'Smooth'
    humidity            = FloatField()      # Environmental humidity %
    temperature         = FloatField()      # Environmental temperature °C
    soil_type           = CharField()       # Categorical: 'Clay', 'Loamy', 'Sandy'
    predicted_label     = CharField()       # Prediction result (Healthy/Disease)
    best_algorithm      = CharField()       # Best performing model used
    created_at          = DateTimeField()   # Timestamp of prediction
```

**Purpose**: Stores historical prediction records for each user, enabling prediction history tracking and model performance analysis.

---

## 🎯 Core Applications

### 1. **BASICS APP** - User Authentication & Banana Leaf Disease Prediction

#### Views (views.py)

##### Authentication Views

- **index()**: Redirects authenticated users to dashboard, others to login
- **signup_view()**: Handles user registration with password validation
- **login_view()**: Authenticates users and creates session
- **logout_view()**: Destroys user session

##### Prediction Views

- **dashboard_view()**: Main page where users choose between Banana Leaf or Air Quality prediction
- **predict_view()**: Core prediction logic for banana leaf disease

#### Models Implemented

The application uses **6 different ML algorithms** for banana leaf disease prediction:

| Algorithm              | File                    | Purpose                    |
| ---------------------- | ----------------------- | -------------------------- |
| Logistic Regression    | logistic_regression.pkl | Linear probability model   |
| Decision Tree          | decision_tree.pkl       | Tree-based decision model  |
| Random Forest          | random_forest.pkl       | Ensemble of decision trees |
| Support Vector Machine | svm.pkl                 | Margin-based classifier    |
| K-Nearest Neighbors    | knn.pkl                 | Instance-based learner     |
| Naive Bayes            | naive_bayes.pkl         | Probabilistic classifier   |

#### Prediction Input Features

The form collects 9 features from the user:

1. **Leaf Length (cm)** - Numeric input [10-80]
2. **Leaf Width (cm)** - Numeric input [5-40]
3. **Colour Intensity** - Categorical dropdown (Dark, Light, Medium)
4. **Spots Present** - Binary toggle (0=No, 1=Yes)
5. **Moisture Level (%)** - Numeric input [20-100]
6. **Texture** - Categorical dropdown (Rough, Smooth)
7. **Humidity (%)** - Numeric input [30-100]
8. **Temperature (°C)** - Numeric input [15-42]
9. **Soil Type** - Categorical dropdown (Clay, Loamy, Sandy)

#### Disease Classes

- Healthy
- Black Sigatoka
- Yellow Sigatoka
- Panama Disease
- Moko Disease

#### Prediction Logic (predict_view)

1. Receives form inputs from user
2. Encodes categorical features using trained LabelEncoders
3. Loads all 6 trained models from saved_models/
4. Runs prediction on each model
5. Selects **best performing model** (highest accuracy)
6. Displays results table showing all predictions & accuracies
7. Highlights best algorithm in orange
8. Stores prediction record in database

#### URL Routes

```
/                   → index
/signup/            → signup_view
/login/             → login_view
/logout/            → logout_view
/dashboard/         → dashboard_view
/predict/           → predict_view
```

---

### 2. **AIRQUALITY APP** - Air Quality Classification

#### Views (views.py)

- **aq_home()**: Landing page for air quality module
- **aq_predict()**: Air quality prediction with same multi-model approach

#### Models Implemented

Same 6 algorithms as banana leaf, trained on air quality data:

- lr_aq.pkl (Logistic Regression)
- dt_aq.pkl (Decision Tree)
- rf_aq.pkl (Random Forest)
- svm_aq.pkl (SVM)
- knn_aq.pkl (KNN)
- nb_aq.pkl (Naive Bayes)

#### Prediction Input Features

The form collects 6 air quality parameters:

1. **CO (Carbon Monoxide)** - Numeric [ppm]
2. **NO2 (Nitrogen Dioxide)** - Numeric [ppb]
3. **SO2 (Sulfur Dioxide)** - Numeric [ppb]
4. **O3 (Ozone)** - Numeric [ppb]
5. **PM2.5 (Fine Particulates)** - Numeric [μg/m³]
6. **PM10 (Coarse Particulates)** - Numeric [μg/m³]

#### Air Quality Classification

The model classifies locations as:

- **Industrial** - High pollution from industrial activities
- **Residential** - Moderate pollution from urban areas

#### URL Routes

```
/airquality/        → aq_home
/airquality/predict/ → aq_predict
```

---

## 🤖 Machine Learning Pipeline

### Training Scripts

#### 1. train_models.py (Banana Leaf Disease)

**Purpose**: Train all 6 ML algorithms on banana leaf dataset

**Process**:

1. Load `banana_leaf_dataset.csv` (9 features, disease labels)
2. Encode categorical features (Texture, SoilType, ColorIntensity)
3. Generate 4000 synthetic training samples using **clear scoring rules**:
   - Score += 3 if no spots
   - Score += 2 if moisture > 55%
   - Score += 2 if humidity < 72%
   - Score += 1 if temperature < 33°C
   - Classify as "Healthy" if score ≥ 5
4. Combine real + synthetic data
5. Split into 80% train, 20% test
6. Train 6 algorithms with StandardScaler preprocessing
7. Calculate accuracy scores via cross-validation
8. Save models and encoders to `saved_models/`

**Output Files**:

- 6 trained .pkl files (models)
- 4 label encoder files (.pkl)
- `accuracies.pkl` (performance metrics)

#### 2. train_airquality.py (Air Quality)

**Purpose**: Train models on air quality dataset

**Process**:

1. Load `airquality_dataset.csv` (6 features: CO, NO2, SO2, O3, PM2.5, PM10)
2. Drop unused columns (Date, City)
3. Encode target (Industrial=0, Residential=1)
4. Create Pipeline: StandardScaler → Model
5. Train 6 algorithms
6. Evaluate on test set
7. Save to `saved_models_aq/`

**Output Files**:

- 6 trained .pkl files
- `label_encoder_aq.pkl`
- `accuracies_aq.pkl`

### Model Comparison Architecture

Both apps implement an **ensemble-like selection strategy**:

1. Load all 6 pre-trained models
2. Get predictions from each model
3. Track accuracy of each model
4. **Select the model with highest accuracy** as final prediction
5. Display results from all models for comparison

This approach provides:

- **Robustness**: Multiple models validate predictions
- **Transparency**: Users see all predictions
- **Reliability**: Best-performing model selected automatically

---

## 🎨 User Interface

### Page Hierarchy

```
HOME (index.html)
    ↓
LOGIN (login.html) or SIGNUP (signup.html)
    ↓
DASHBOARD (dashboard.html)
    ├→ Banana Leaf Disease Prediction
    │   └→ PREDICT (predict.html)
    │       └→ Results with model comparison
    │
    └→ Air Quality Prediction
        └→ AQ_PREDICT (aq_predict.html)
            └→ Results with model comparison
```

### Design System

**Color Scheme**:

- Primary Orange: `#F5A623`
- Dark Navy: `#1a1a1a`
- Light Cream: `#FFFDF7`
- Secondary Blue: `#2196F3`

**Typography**:

- Heading Font: 'Syne' (bold, geometric)
- Body Font: 'Inter' (clean, readable)

**Layout**:

- Responsive grid layouts
- Cards with shadow effects
- Smooth animations and transitions
- Mobile-first design approach

---

## 🔐 Authentication System

### User Registration (signup_view)

- Validates matching passwords
- Checks for duplicate usernames
- Creates Django User object
- Redirects to login

### User Login (login_view)

- Authenticates username/password
- Creates session on success
- Shows error messages on failure
- Redirects to dashboard

### Protected Routes

All prediction views require login:

```python
@login_required(login_url='/login/')
def predict_view(request):
    # Only logged-in users can access
```

---

## 🚀 Key Features

### 1. **Multi-Model Ensemble**

- 6 different algorithms trained on same data
- All models run simultaneously on input
- Best model selected automatically
- Full comparison table displayed

### 2. **User Authentication**

- Secure registration and login
- Session-based authentication
- Prediction history linked to users
- Logout functionality

### 3. **Data Encoding**

- Categorical features encoded using LabelEncoders
- Encoders saved and loaded dynamically
- Transparent encoding/decoding in views

### 4. **Dual Prediction Domains**

- Banana Leaf Disease (5 categories)
- Air Quality Type (2 categories)
- Completely separate pipelines
- Independent model training

### 5. **Responsive Design**

- Mobile-friendly interface
- Adaptive grid layouts
- Touch-friendly form inputs
- Cross-browser compatible

### 6. **Model Persistence**

- All models saved as pickle files
- Encoders packaged with models
- No retraining needed on startup
- Fast prediction inference

---

## 📝 Data Flow

### Banana Leaf Prediction Flow

```
User Input Form
    ↓
Encode Categorical Values
    ├→ Texture → le_texture.pkl → numeric code
    ├→ SoilType → le_soil.pkl → numeric code
    └→ ColorIntensity → le_color.pkl → numeric code
    ↓
Create Feature Array [9 features]
    ↓
Load 6 Models from saved_models/
    ├→ logistic_regression.pkl → predict → label + accuracy
    ├→ decision_tree.pkl → predict → label + accuracy
    ├→ random_forest.pkl → predict → label + accuracy
    ├→ svm.pkl → predict → label + accuracy
    ├→ knn.pkl → predict → label + accuracy
    └→ naive_bayes.pkl → predict → label + accuracy
    ↓
Decode Numeric Predictions
    └→ Use label_encoder.pkl to convert [0-4] → disease name
    ↓
Find Best Model (highest accuracy)
    ↓
Display Results
    ├→ Final Prediction (from best model)
    ├→ Best Algorithm Name
    ├→ Confidence Score
    └→ Comparison Table of All Models
    ↓
Save PredictionRecord to Database
```

### Air Quality Prediction Flow

```
User Input Form (6 pollutant levels)
    ↓
Create Feature Array [6 features]
    ↓
Load 6 Models from saved_models_aq/
    ├→ lr_aq.pkl → predict → label + accuracy
    ├→ dt_aq.pkl → predict → label + accuracy
    ├→ rf_aq.pkl → predict → label + accuracy
    ├→ svm_aq.pkl → predict → label + accuracy
    ├→ knn_aq.pkl → predict → label + accuracy
    └→ nb_aq.pkl → predict → label + accuracy
    ↓
Decode Predictions
    └→ 0="Industrial" / 1="Residential"
    ↓
Select Best Model
    ↓
Display Results & Comparison Table
```

---

## 🛠️ Setup & Installation

### Prerequisites

- Python 3.8+
- pip package manager
- Virtual environment (recommended)

### Installation Steps

```bash
# 1. Navigate to project directory
cd Django_PML6/GUI

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 4. Install dependencies
pip install django==4.2.25
pip install scikit-learn pandas numpy joblib

# 5. Run migrations
python manage.py migrate

# 6. Create superuser (admin)
python manage.py createsuperuser

# 7. Start development server
python manage.py runserver
```

### Running ML Training

```bash
# Train banana leaf models
python ml/train_models.py

# Train air quality models
python ml/train_airquality.py
```

---

## 📚 Important Files Reference

| File                     | Purpose                                        |
| ------------------------ | ---------------------------------------------- |
| `settings.py`            | Django configuration, installed apps, database |
| `urls.py` (main)         | Global URL routing                             |
| `basics/urls.py`         | Authentication & banana prediction routes      |
| `airquality/urls.py`     | Air quality prediction routes                  |
| `basics/views.py`        | Authentication & prediction logic              |
| `airquality/views.py`    | Air quality prediction logic                   |
| `basics/models.py`       | PredictionRecord database model                |
| `manage.py`              | Django CLI tool                                |
| `ml/train_models.py`     | Banana leaf model training                     |
| `ml/train_airquality.py` | Air quality model training                     |
| `retrain_models.py`      | Re-train all models (main script)              |

---

## 🔍 Model Performance

### Banana Leaf Models

Accuracies stored in `ml/saved_models/accuracies.pkl`:

- Contains performance scores for each algorithm
- Loaded and displayed in prediction results
- Used to select best-performing model

### Air Quality Models

Accuracies stored in `ml/saved_models_aq/accuracies_aq.pkl`:

- Performance metrics for each algorithm
- Displayed in prediction results table

---

## 📊 Data Formats

### Input CSV Formats

**banana_leaf_dataset.csv**:

```
LeafLength(cm), LeafWidth(cm), ColorIntensity, SpotsPresent, MoistureLevel(%),
Texture, Humidity(%), Temperature(°C), SoilType, DiseaseLabel
10.5, 8.3, Dark, 0, 65.2, Smooth, 72.1, 28.5, Loamy, Healthy
...
```

**airquality_dataset.csv**:

```
CO, NO2, SO2, O3, PM2.5, PM10, Type, City, Date
2.5, 45.3, 12.1, 65.8, 35.2, 120.5, Industrial, CityA, 2024-01-01
...
```

---

## 🐛 Error Handling

### In predict_view

```python
try:
    # Process inputs and predictions
except Exception as e:
    context['error'] = f'Prediction failed: {str(e)}'
    return render(request, 'predict.html', context)
```

### Model Loading

- Checks file existence before loading
- Returns None if model not found
- Graceful degradation if encoders missing

---

## 🎓 Educational Value

This project demonstrates:

1. **Full-Stack Development**
   - Backend: Django framework, Python
   - Frontend: HTML/CSS/JavaScript
   - Database: Django ORM with SQLite

2. **Machine Learning Integration**
   - Model training pipeline
   - Feature encoding/decoding
   - Ensemble prediction strategy
   - Model persistence with joblib

3. **Software Engineering Best Practices**
   - Separation of concerns (apps, views, models)
   - DRY principle (reusable encoders)
   - Authentication & authorization
   - Error handling & validation

4. **User Experience Design**
   - Responsive layouts
   - Clear visual hierarchy
   - Intuitive form design
   - Immediate feedback

---

## 📝 Future Enhancement Opportunities

1. **Advanced Features**
   - Real-time model retraining
   - User feedback loop for model improvement
   - Model versioning system
   - A/B testing different models

2. **Performance**
   - Caching model predictions
   - Asynchronous model loading
   - Database query optimization
   - API rate limiting

3. **ML Improvements**
   - Hyperparameter tuning
   - Cross-validation scoring
   - Feature importance analysis
   - Confusion matrix visualization

4. **UI/UX**
   - Dark mode support
   - Prediction history charts
   - Export results (CSV/PDF)
   - Real-time model comparison charts

5. **Security**
   - CSRF protection (already implemented)
   - Rate limiting on predictions
   - Input validation hardening
   - SQL injection prevention

---

## 📞 Contact & Support

For questions about this project:

- Review Django documentation: https://docs.djangoproject.com/
- Review scikit-learn: https://scikit-learn.org/
- Check inline code comments in views.py and models.py

---

## 📄 License

This is a educational project demonstrating Django + ML integration.

---

**Document Generated**: May 8, 2026
**Project Status**: Complete & Functional
**Django Version**: 4.2.25
**Python Version**: 3.x
