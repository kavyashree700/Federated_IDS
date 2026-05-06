# Federated Intrusion Detection System (IDS)

## 📌 Project Overview

This project builds a machine learning-based Intrusion Detection System (IDS) using network traffic data from the CICIDS2017 dataset.

The system is developed in multiple phases, starting from basic preprocessing to advanced attack classification.

---

## 📊 Dataset

* Source: CICIDS2017
* Contains normal and malicious network traffic
* Includes multiple attack types like:

  * DDoS
  * PortScan
  * Bot
  * Web Attacks

---

## ⚙️ Project Phases

### 🔹 Phase 1: Data Preprocessing

* Loaded dataset (Friday dataset)
* Removed duplicate records
* Handled missing and infinite values
* Converted labels:

  * 0 → BENIGN
  * 1 → ATTACK

---

### 🔹 Phase 2: Dataset Combination

* Combined multiple datasets (Monday, Tuesday, Friday)
* Cleaned merged dataset
* Maintained binary classification

---

### 🔜 Phase 3 (Upcoming)

* Multi-class classification
* Detect different attack types separately

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn

---

## 📂 Project Structure

federated_ids/
│
├── data/ (ignored in GitHub)
├── src/
│   ├── data_preprocessing.py
│   ├── combine_datasets.py
│
├── notebooks/
├── README.md
└── .gitignore

---

## 🚀 How to Run

1. Install dependencies:
   pip install pandas numpy scikit-learn

2. Run preprocessing:
   python src/data_preprocessing.py

3. Combine datasets:
   python src/combine_datasets.py

---

## 🎯 Goal

To build an intelligent intrusion detection system capable of identifying malicious network activities efficiently.
