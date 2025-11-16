# 🌾 **Crop Rotation ML System – Production Ready**

### *AI-powered Crop Recommendation Engine for Smart Indian Agriculture* 🇮🇳🤖

---

## ⭐ **Overview**

The **Crop Rotation ML System** is an advanced AI-driven platform that recommends the best next crop for farmers based on soil type, weather, market conditions, and historical crop patterns. Designed especially for **Indian agriculture**, this system boosts yield, improves soil health, and increases farmer income.

---

## 🚀 **Core Features**

| Feature                              | Description                                        |
| ------------------------------------ | -------------------------------------------------- |
| 🤖 **ML-Based Crop Recommendation**  | Achieves **95%+ accuracy** using trained models    |
| ☁️ **Real-Time Weather Integration** | Uses OpenWeather API for accurate predictions      |
| 🌱 **Soil Type Analysis**            | Supports major Indian soil categories              |
| 📈 **Market Trend Analysis**         | Considers price fluctuations & demand              |
| 🌾 **20+ Supported Crops**           | Wheat, Rice, Maize, Bajra, Sugarcane, Pulses, etc. |
| ⚡ **FastAPI Backend**                | Ultra-fast & production-ready                      |

---

## 🛠️ **Tech Stack**

| Category          | Technology                         |
| ----------------- | ---------------------------------- |
| Backend Framework | ⚡ FastAPI 0.104+                   |
| ML Libraries      | 🧠 Scikit-learn 1.3+, XGBoost 2.0+ |
| Language          | 🐍 Python 3.11+                    |
| Deployment        | 🚀 Render / Railway / AWS EC2      |
| Weather API       | 🌤️ OpenWeather API                |

---

## 📦 **Installation & Quick Start**

### 🔽 1. Clone the Repository

```bash
git clone https://github.com/your-username/crop-rotation-ml.git
cd crop-rotation-ml
```

---

### 📌 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```

---

### 📥 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### ▶️ 4. Run the FastAPI Server

```bash
uvicorn app.main:app --reload
```

Server will start at:

```
http://127.0.0.1:8000
```

Interactive API Docs (Swagger UI):
👉 **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

---

## 🔍 **API Endpoints**

| Endpoint            | Method | Description                   |
| ------------------- | ------ | ----------------------------- |
| `/predict-rotation` | POST   | Get recommended next crop     |
| `/weather/{city}`   | GET    | Fetch real-time weather       |
| `/soil-types`       | GET    | List all supported soil types |
| `/crops`            | GET    | List all supported crops      |
| `/health`           | GET    | Server health check           |

---

## 📊 **Sample Request (Crop Rotation Prediction)**

```json
{
  "current_crop": "Wheat",
  "soil_type": "Loamy",
  "rainfall_mm": 120,
  "temperature": 32,
  "state": "Uttar Pradesh"
}
```

---

## 🧠 **Machine Learning Workflow**

1. 📥 Data collection (soil + weather + crop datasets)
2. 🧹 Preprocessing & feature engineering
3. 🔧 Model training (RandomForest, XGBoost)
4. 🏆 Best model selection (Accuracy > 95%)
5. 📤 API integration via FastAPI
6. 🚀 Production deployment

---

## 🛰️ **Live Demo**

👉 `https://your-deployed-api-url.com`

---

## 📁 **Project Structure**

```
📦 crop-rotation-ml
├── 📂 app
│   ├── main.py
│   ├── routes
│   ├── models
│   ├── ml
│   └── utils
├── requirements.txt
├── README.md
└── model.pkl
```

---

## 🤝 **Contributing**

Contributions are always welcome! Feel free to open **Issues** or **Pull Requests**.

---

## 📞 **Contact**

**Developer:** Somveer Singh
📧 Email: [kaidwal.somveer@gmail.com](mailto:kaidwal.somveer@gmail.com)
🌐 LinkedIn: [https://www.linkedin.com/in/somveer-singh-0205971ab/](https://www.linkedin.com/in/somveer-singh-0205971ab/)

---

## ⭐ **Support the Project**
If you like this project, give it a **⭐ on GitHub** — it motivates the development!
