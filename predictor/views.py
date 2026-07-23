from xml.parsers.expat import model
from django.shortcuts import get_object_or_404, redirect, render, HttpResponse
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from predictor.models import LungCancerPrediction
from django.contrib.auth.decorators import login_required
from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from .models import LungCancerPrediction

@login_required(login_url="login")
def home(request):
    return render(request, "home.html")

@login_required(login_url="login")
def predict(request):
    # ================== LOAD DATASET ==================
    data = pd.read_csv("C:/Users/user/Downloads/Project_Dataset_2026/lungcancerdataset/survey_lung_cancer.csv")

    data['GENDER'] = data['GENDER'].map({'M': 1, 'F': 0})
    data['LUNG_CANCER'] = data['LUNG_CANCER'].map({'YES': 1, 'NO': 0})
    for col in data.columns:
        if col not in ['GENDER', 'LUNG_CANCER', 'AGE']:
            data[col] = data[col].map({1: 0, 2: 1})


    X = data.drop('LUNG_CANCER', axis=1)
    y = data['LUNG_CANCER']

    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

    # ================== MODELS ==================
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC()
    }

    best_model = None
    best_accuracy = 0
    accuracies = {}

    # ================== TRAIN & COMPARE ==================
    for name, model in models.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)

        acc = accuracy_score(y_test, pred)
        accuracies[name] = round(acc * 100, 2)

        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_model_name = name

    # ================== HANDLE FORM ==================
    if request.method == "POST":

        sample = pd.DataFrame([{
            'GENDER': int(request.POST['gender']),
            'AGE': int(request.POST['age']),
            'SMOKING': int(request.POST['smoking']),
            'YELLOW_FINGERS': int(request.POST['yellow_fingers']),
            'ANXIETY': int(request.POST['anxiety']),
            'PEER_PRESSURE': int(request.POST['peer_pressure']),
            'CHRONIC_DISEASE': int(request.POST['chronic_disease']),
            'FATIGUE': int(request.POST['fatigue']),
            'ALLERGY': int(request.POST['allergy']),
            'WHEEZING': int(request.POST['wheezing']),
            'ALCOHOL_CONSUMING': int(request.POST['alcohol']),
            'COUGHING': int(request.POST['coughing']),
            'SHORTNESS_OF_BREATH': int(request.POST['short_breath']),
            'SWALLOWING_DIFFICULTY': int(request.POST['swallowing']),
            'CHEST_PAIN': int(request.POST['chest_pain'])
        }])

        result = best_model.predict(sample)

        if result[0] == 1:
            output = "High Risk of Lung Cancer"
        else:
            output = "Low Risk"

        # Save to DB
        LungCancerPrediction.objects.create(
            gender=sample['GENDER'][0],
            age=sample['AGE'][0],
            smoking=sample['SMOKING'][0],
            yellow_fingers=sample['YELLOW_FINGERS'][0],
            anxiety=sample['ANXIETY'][0],
            peer_pressure=sample['PEER_PRESSURE'][0],
            chronic_disease=sample['CHRONIC_DISEASE'][0],
            fatigue=sample['FATIGUE'][0],
            allergy=sample['ALLERGY'][0],
            wheezing=sample['WHEEZING'][0],
            alcohol=sample['ALCOHOL_CONSUMING'][0],
            coughing=sample['COUGHING'][0],
            short_breath=sample['SHORTNESS_OF_BREATH'][0],
            swallowing=sample['SWALLOWING_DIFFICULTY'][0],
            chest_pain=sample['CHEST_PAIN'][0],
            result=output
        )

        return render(request, "result.html", {
            "result": output,
            "submitted": True,
            "best_model": best_model_name,
            "accuracies": accuracies
        })

    return render(request, "predict.html", {
        "accuracies": accuracies,
        "best_model": best_model_name
    })

@login_required(login_url="login")
def model_comparison(request):
    import pandas as pd

    path = "C:\\Users\\user\\Downloads\\Project_Dataset_2026\\lungcancerdataset\\survey_lung_cancer.csv"
    data = pd.read_csv(path)

    # preprocessing
    data['GENDER'] = data['GENDER'].map({'M': 1, 'F': 0})
    data['LUNG_CANCER'] = data['LUNG_CANCER'].map({'YES': 1, 'NO': 0})

    X = data.drop('LUNG_CANCER', axis=1)
    y = data['LUNG_CANCER']

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=42
    )

    from sklearn.metrics import accuracy_score

    # Logistic Regression
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    acc_lr = accuracy_score(y_test, lr.predict(X_test)) * 100

    # Decision Tree
    from sklearn.tree import DecisionTreeClassifier
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    acc_dt = accuracy_score(y_test, dt.predict(X_test)) * 100

    # Random Forest
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    acc_rf = accuracy_score(y_test, rf.predict(X_test)) * 100

    # SVM
    from sklearn.svm import SVC
    svm = SVC()
    svm.fit(X_train, y_train)
    acc_svm = accuracy_score(y_test, svm.predict(X_test)) * 100

    context = {
        "lr": round(acc_lr, 2),
        "dt": round(acc_dt, 2),
        "rf": round(acc_rf, 2),
        "svm": round(acc_svm, 2),
    }

    return render(request, "comparison.html", context)

@login_required(login_url="login")
def about(request):
    return render(request, 'about.html')

@login_required(login_url="login")
def history_view(request):
    query = request.GET.get('q')

    history = LungCancerPrediction.objects.all().order_by('-id')

    if query:
        if query.lower() == "positive":
            history = history.filter(result="High Risk of Lung Cancer")
        elif query.lower() == "negative":
            history = history.filter(result="Low Risk")

    return render(request, "history.html", {
        "history": history
    })

def signup(request):
    if(request.method == "POST"):
        uname = request.POST.get("username")
        email = request.POST.get("email")
        password1 = request.POST.get("password1")
        password2 = request.POST.get("password2")
        if password1 != password2:
            return HttpResponse("Passwords do not match")
        else:
            my_user = User.objects.create_user(username=uname, email=email, password=password1)
            my_user.save()
            return redirect("login")
    return render(request, "signup.html")

def user_login(request):
    if(request.method == "POST"):
        uname = request.POST.get("username")
        password = request.POST.get("pass")
        user = User.objects.filter(username = uname)
        user = authenticate(request, username=uname, password=password)
        if user is not None:
            login(request, user)
            return redirect("home")
        else:
            result = "Invalid username or password"
            return HttpResponse("username or password is incorrect")
    return render(request, "login.html")

def user_logout(request):
    logout(request)
    return redirect('login')

def delete_record(request, id):
    record = get_object_or_404(LungCancerPrediction, id=id)
    record.delete()  
    return redirect('history')  

@login_required(login_url="login")
def lungcare(request):
    return render(request, 'lungcare.html')