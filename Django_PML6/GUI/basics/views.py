from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.contrib import messages
import numpy as np
import joblib
import os

# ── Model paths ──────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'ml', 'saved_models')

def get_label(pred):
    """Decode numeric prediction back to original label string"""
    le_path = os.path.join(MODEL_DIR, 'label_encoder.pkl')
    if os.path.exists(le_path):
        le = joblib.load(le_path)
        return le.inverse_transform([int(pred)])[0]
    return str(pred)

ALGORITHMS = {
    'Logistic Regression':       'logistic_regression.pkl',
    'Decision Tree':             'decision_tree.pkl',
    'Random Forest':             'random_forest.pkl',
    'Support Vector Machine':    'svm.pkl',
    'K-Nearest Neighbors':       'knn.pkl',
    'Naive Bayes':               'naive_bayes.pkl',
}

LABEL_MAP = {
    0: 'Healthy',
    1: 'Black Sigatoka',
    2: 'Yellow Sigatoka',
    3: 'Panama Disease',
    4: 'Moko Disease',
}

def load_model(filename):
    path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(path):
        return joblib.load(path)
    return None

# ── Pages ─────────────────────────────────────────────────────────

def index(request):
    """Redirect to dashboard if logged in, otherwise to login"""
    if request.user.is_authenticated:
        return redirect('dashboard')
    else:
        return redirect('login')



def signup_view(request):
    if request.method == 'POST':
        username  = request.POST['username']
        email     = request.POST['email']
        password1 = request.POST['password1']
        password2 = request.POST['password2']

        if password1 != password2:
            return render(request, 'signup.html', {'error': 'Passwords do not match'})

        if User.objects.filter(username=username).exists():
            return render(request, 'signup.html', {'error': 'Username already taken'})

        user = User.objects.create_user(username=username, email=email, password=password1)
        user.save()
        return redirect('login')  # or wherever you want to send them after signup

    return render(request, 'signup.html')

def login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            return redirect('dashboard')
        else:
            messages.error(request, 'Invalid username or password.')

    return render(request, 'login.html')

def logout_view(request):
    logout(request)
    return redirect('login')

@login_required(login_url='/login/')
def dashboard_view(request):
    """Dashboard to choose between Banana Leaf Disease or Air Quality predictions"""
    return render(request, 'dashboard.html')

@login_required(login_url='/login/')
def predict_view(request):
    context = {}

    if request.method == 'POST':
        try:
            # ── Read inputs ──────────────────────────────────────
            leaf_length      = float(request.POST['leaf_length'])
            leaf_width       = float(request.POST['leaf_width'])
            colour_intensity_raw = request.POST['colour_intensity']
            le_color = joblib.load(os.path.join(MODEL_DIR, 'le_color.pkl'))
            colour_intensity = int(le_color.transform([colour_intensity_raw])[0])  
            spots_present    = int(request.POST['spots_present'])      # 0/1
            moisture_level   = float(request.POST['moisture_level'])
            texture_raw  = request.POST['texture']
            soil_raw     = request.POST['soil_type']

            le_texture = joblib.load(os.path.join(MODEL_DIR, 'le_texture.pkl'))
            le_soil    = joblib.load(os.path.join(MODEL_DIR, 'le_soil.pkl'))

            texture   = int(le_texture.transform([texture_raw])[0])
            soil_type = int(le_soil.transform([soil_raw])[0])        # encoded 0-2
            humidity         = float(request.POST['humidity'])
            temperature      = float(request.POST['temperature'])
                    # encoded 0-2

            features = np.array([[
    leaf_length, leaf_width, colour_intensity,
    spots_present, moisture_level, texture,
    humidity, temperature, soil_type]])

            # ── Run all models ───────────────────────────────────
            results = {}
            best_algo = None
            best_acc  = -1

            # Load accuracy scores saved during training
            acc_path = os.path.join(MODEL_DIR, 'accuracies.pkl')
            accuracies = joblib.load(acc_path) if os.path.exists(acc_path) else {}

            for algo_name, filename in ALGORITHMS.items():
                model = load_model(filename)
                if model:
                    pred = model.predict(features)[0]
                    label = get_label(pred)
                    acc   = round(accuracies.get(algo_name, 0) * 100, 2)
                    results[algo_name] = {'label': label, 'accuracy': acc}

                    if acc > best_acc:
                        best_acc  = acc
                        best_algo = algo_name

            # Prediction from best model
            best_model = load_model(ALGORITHMS[best_algo])
            final_pred = best_model.predict(features)[0]
            final_label = get_label(final_pred)

            context = {
                'results':     results,
                'best_algo':   best_algo,
                'best_acc':    best_acc,
                'final_label': final_label,
                'submitted':   True,
            }

        except Exception as e:
            context['error'] = f'Prediction failed: {str(e)}'

    return render(request, 'predict.html', context)