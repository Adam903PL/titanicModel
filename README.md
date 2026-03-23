# 🚢 Titanic Model – jak to działało

## 💡 Główna idea

Model miał jedno zadanie:

> przewidzieć, czy pasażer Titanica przeżył (1) czy nie (0)

To jest tzw. **binary classification** ([Medium][1])

---

## 📊 Dane wejściowe (dataset)

Model korzysta z datasetu z Kaggle:

👉 zawiera dane o pasażerach, np.:

* wiek (`Age`)
* płeć (`Sex`)
* klasa (`Pclass`)
* cena biletu (`Fare`)
* port wejścia (`Embarked`)
* itd.

Celem było znalezienie zależności między tymi cechami a przeżyciem ([GeeksforGeeks][2])

---

## 🧠 Jak działa model (pipeline)

Typowy pipeline (i prawie na pewno Twój też):

### 1. Data loading

```python
df = pd.read_csv("train.csv")
```

Masz:

* train dataset → z odpowiedziami (Survived)
* test dataset → bez odpowiedzi

---

### 2. Data preprocessing

Najważniejszy etap:

* uzupełnianie braków (np. Age)
* zamiana tekstów na liczby:

  * male → 0
  * female → 1
* encoding kategorii (`Embarked`, `Pclass`)

👉 bez tego model nie działa

---

### 3. Feature engineering

Czyli poprawianie danych:

* tworzenie nowych cech
* np.:

  * czy ktoś podróżował sam
  * title z imienia (Mr, Mrs itd.)

To mega boostuje accuracy.

---

### 4. Trenowanie modelu

Najczęściej używane modele:

* Logistic Regression
* Random Forest
* SVM
* XGBoost ([ResearchGate][3])

Bardzo często beginnerzy używają:

```python
from sklearn.ensemble import RandomForestClassifier
```

---

### 5. Prediction

Model robi:

```python
model.predict(X_test)
```

I zwraca:

```
0 → dead
1 → survived
```

---

### 6. Evaluation

Na Kaggle liczy się:

> % poprawnych odpowiedzi (accuracy) ([Dataquest][4])

---

## 🔥 Co model faktycznie “nauczył się”

Najważniejsze wnioski:

* kobiety miały dużo większe szanse przeżyć
* dzieci też
* klasa (1st class > 3rd class) była kluczowa

👉 np. płeć była NAJWAŻNIEJSZA ([Medium][5])

---

## 🧩 Jak to opisać w README (gotowiec)

Możesz wrzucić coś takiego:

---

# 🚢 Titanic Model – Survival Prediction

This project is a machine learning model built on the **Kaggle Titanic dataset**, designed to predict whether a passenger survived the disaster.

## 💡 Overview

The model performs **binary classification**, predicting:

* `1` → survived
* `0` → did not survive

based on passenger data such as age, gender, and ticket class.

---

## 🧠 How It Works

The model follows a standard ML pipeline:

1. Data loading (train/test datasets)
2. Data preprocessing (handling missing values, encoding features)
3. Feature engineering
4. Model training (e.g. Random Forest / Logistic Regression)
5. Prediction on unseen data

---

## 📊 Features Used

* Age
* Sex
* Passenger class (Pclass)
* Fare
* Embarked

---

## 🔍 Key Insights

* Gender was one of the most important factors
* Higher-class passengers had better survival rates
* Children had higher survival probability

---

## 🧠 What This Project Demonstrates

* Machine learning fundamentals
* Data preprocessing and feature engineering
* Classification models
* Working with real-world datasets

---

## 🧩 Summary

This project is a classic beginner ML task that demonstrates how to build a predictive model from real-world data and evaluate its performance.

---

## 💬 TL;DR dla Ciebie

Twój projekt to:

👉 **klasyczny ML classifier**
👉 uczony na danych pasażerów
👉 przewidujący przeżycie

---
