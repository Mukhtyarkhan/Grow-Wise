# 🌾 GrowWise — AI Agricultural Platform

> Final Year Project · Grade: A · Team Lead (4 members) · CNN 90% accuracy · SVM 98% accuracy

[![Flask](https://img.shields.io/badge/Backend-Flask-000000?style=for-the-badge&logo=flask)](https://flask.palletsprojects.com/)
[![TensorFlow](https://img.shields.io/badge/AI-TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow)](https://tensorflow.org/)
[![Scikit-learn](https://img.shields.io/badge/ML-Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org/)
[![MySQL](https://img.shields.io/badge/Database-MySQL-4479A1?style=for-the-badge&logo=mysql)](https://www.mysql.com/)

---

## 🎯 Problem Solved

Pakistan and many developing nations lose significant crop yields annually due to undetected plant diseases and poor crop-soil matching. Farmers lack affordable access to agronomic expertise. GrowWise democratizes that knowledge using AI — available through a simple web interface, no agronomist required.

**Outcome: Grade A · 2 production-ready AI models · REST API serving 100+ test users**

---

## 📊 Model Performance

| Model | Task | Algorithm | Accuracy |
|-------|------|-----------|----------|
| Disease Detector | Identify plant diseases from leaf images | CNN (TensorFlow) | **90%** |
| Crop Recommender | Recommend optimal crops for soil/climate | SVM (Scikit-learn) | **98%** |

---

## ✨ Features

### 🔬 Plant Disease Detection (CNN)
- User uploads a leaf image
- CNN model classifies the disease from 38+ disease categories
- Returns: disease name, confidence score, treatment recommendations
- Model trained on PlantVillage dataset (54,000+ images)

### 🌱 Crop Recommendation Engine (SVM)
- User inputs soil parameters (N, P, K, pH, humidity, temperature, rainfall)
- SVM model recommends the optimal crop for those conditions
- Returns top 3 crop recommendations with confidence scores
- 98% test accuracy across 22 crop categories

### 🌐 REST API
- Flask-based REST API
- JSON responses for both AI models
- Designed for mobile app integration
- Documented with Swagger/OpenAPI
- Served 100+ users during testing phase

### 📊 Analytics Dashboard
- Aggregated disease detection trends by region
- Crop recommendation history per user
- Admin panel for model performance monitoring

---

## 🏗️ System Architecture

```
User / Mobile App
      │
      │ HTTP / REST API
      ▼
┌─────────────────────────────────────────┐
│           Flask REST Backend            │
│                                         │
│  ┌─────────────────┐  ┌──────────────┐  │
│  │  Image Upload   │  │  Soil Input  │  │
│  │  Endpoint       │  │  Endpoint    │  │
│  └───────┬─────────┘  └──────┬───────┘  │
│          │                   │           │
│  ┌───────▼──────┐   ┌────────▼────────┐ │
│  │  CNN Model   │   │   SVM Model     │ │
│  │ (TensorFlow) │   │ (Scikit-learn)  │ │
│  │  .h5 file    │   │  .pkl file      │ │
│  └───────┬──────┘   └────────┬────────┘ │
│          │                   │           │
│  ┌───────▼───────────────────▼────────┐ │
│  │              MySQL DB              │ │
│  │  (Users · Predictions · Analytics) │ │
│  └────────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

---

## 🧠 CNN Architecture (Disease Detection)

```python
model = Sequential([
    # Feature extraction layers
    Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2, 2),

    # Classification layers
    Flatten(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dense(38, activation='softmax')  # 38 disease categories
])

# Training config
model.compile(
    optimizer=Adam(lr=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
# Result: 90% validation accuracy
```

---

## 🧠 SVM Pipeline (Crop Recommendation)

```python
# Feature engineering
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
X = df[features]
y = df['label']  # 22 crop types

# Pipeline with scaling + SVM
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(
        kernel='rbf',
        C=10,
        gamma='scale',
        probability=True  # for confidence scores
    ))
])

pipeline.fit(X_train, y_train)
# Result: 98% test accuracy
```

---

## 🚀 API Endpoints

```
POST /api/disease/predict
  Body: { image: <base64> }
  Returns: { disease: "Tomato Late Blight", confidence: 0.94, treatment: "..." }

POST /api/crop/recommend
  Body: { N: 90, P: 42, K: 43, temperature: 21, humidity: 82, ph: 6.5, rainfall: 202 }
  Returns: { recommendations: [{ crop: "Rice", confidence: 0.97 }, ...] }

GET /api/crops
  Returns: List of all 22 supported crops

GET /api/diseases
  Returns: List of all 38 detectable diseases
```

---

## 📁 Project Structure

```
growwise/
├── models/
│   ├── cnn/
│   │   ├── train_cnn.py        # Training script
│   │   ├── evaluate.py         # Evaluation metrics
│   │   └── plant_disease.h5    # Trained model
│   └── svm/
│       ├── train_svm.py
│       └── crop_model.pkl      # Trained + serialized model
├── api/
│   ├── app.py                  # Flask application
│   ├── routes/
│   │   ├── disease.py
│   │   └── crop.py
│   └── utils/
│       ├── image_preprocessing.py
│       └── response_formatter.py
├── data/
│   ├── crop_recommendation.csv
│   └── disease_info.json       # Treatment recommendations
├── notebooks/
│   ├── CNN_Training.ipynb
│   └── SVM_Experiments.ipynb
└── requirements.txt
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | Python, Flask |
| CNN | TensorFlow/Keras, OpenCV, NumPy |
| SVM | Scikit-learn, Pandas, NumPy |
| Database | MySQL |
| API Docs | Swagger / OpenAPI |
| Data | PlantVillage dataset, Kaggle crop dataset |

---

## 🏆 Project Outcome

- ✅ **Grade A** — University of Sindh Jamshoro
- ✅ **Team Lead** — managed 4-person team through 6-month development
- ✅ **2 production AI models** with publication-grade accuracy
- ✅ **REST API** serving real users during live testing

---

## 👤 Developer

**Mukhtyar Khan** — Lead Developer & Team Lead
- 💼 [linkedin.com/in/mukhtyar-khan](https://linkedin.com/in/mukhtyar-khan)
- 🌐 [mukhtyar-portfolio.vercel.app](https://mukhtyar-portfolio.vercel.app)
- 📧 [MukhtarKhani382@gmail.com](mailto:MukhtarKhani382@gmail.com)
