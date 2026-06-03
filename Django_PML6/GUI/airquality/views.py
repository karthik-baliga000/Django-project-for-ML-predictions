from django.shortcuts import render
from django.contrib.auth.decorators import login_required
import numpy as np
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'ml', 'saved_models_aq')

ALGORITHMS = {
    'Logistic Regression':    'lr_aq.pkl',
    'Decision Tree':          'dt_aq.pkl',
    'Random Forest':          'rf_aq.pkl',
    'Support Vector Machine': 'svm_aq.pkl',
    'K-Nearest Neighbors':    'knn_aq.pkl',
    'Naive Bayes':            'nb_aq.pkl',
}

def get_label(pred):
    le_path = os.path.join(MODEL_DIR, 'label_encoder_aq.pkl')
    if os.path.exists(le_path):
        le = joblib.load(le_path)
        return le.inverse_transform([int(pred)])[0]
    return str(pred)

@login_required(login_url='/login/')
def aq_home(request):
    return render(request, 'airquality/aq_home.html')

@login_required(login_url='/login/')
def aq_predict(request):
    context = {}

    if request.method == 'POST':
        try:
            # Read inputs
            co   = float(request.POST['co'])
            no2  = float(request.POST['no2'])
            so2  = float(request.POST['so2'])
            o3   = float(request.POST['o3'])
            pm25 = float(request.POST['pm25'])
            pm10 = float(request.POST['pm10'])

            features = np.array([[co, no2, so2, o3, pm25, pm10]])

            # Load accuracies
            acc_path   = os.path.join(MODEL_DIR, 'accuracies_aq.pkl')
            accuracies = joblib.load(acc_path) if os.path.exists(acc_path) else {}

            results   = {}
            best_algo = None
            best_acc  = -1

            for algo_name, filename in ALGORITHMS.items():
                path = os.path.join(MODEL_DIR, filename)
                if os.path.exists(path):
                    model = joblib.load(path)
                    pred  = model.predict(features)[0]
                    label = get_label(pred)
                    acc   = round(accuracies.get(algo_name, 0) * 100, 2)
                    results[algo_name] = {'label': label, 'accuracy': acc}

                    if acc > best_acc:
                        best_acc  = acc
                        best_algo = algo_name

            best_model  = joblib.load(os.path.join(MODEL_DIR, ALGORITHMS[best_algo]))
            final_pred  = best_model.predict(features)[0]
            final_label = get_label(final_pred)

            context = {
                'results':     results,
                'best_algo':   best_algo,
                'best_acc':    best_acc,
                'final_label': final_label,
                'submitted':   True,
                # Pass inputs back to form
                'co':   co,   'no2': no2,
                'so2':  so2,  'o3':  o3,
                'pm25': pm25, 'pm10': pm10,
            }

        except Exception as e:
            context['error'] = f'Prediction failed: {str(e)}'

    return render(request, 'airquality/aq_predict.html', context)