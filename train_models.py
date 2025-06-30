import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib

# 1️⃣ Load dataset bersih
df = pd.read_csv("Sleep_dataset_cleaned.csv")

# 2️⃣ Pisahkan Blood Pressure ke systolic & diastolic
bp_split = df['Blood Pressure'].str.split('/', expand=True)
df['Systolic'] = pd.to_numeric(bp_split[0], errors='coerce')
df['Diastolic'] = pd.to_numeric(bp_split[1], errors='coerce')
df.drop(columns='Blood Pressure', inplace=True)

# 3️⃣ Encode fitur kategorikal
gender_enc = LabelEncoder()
df['Gender_num'] = gender_enc.fit_transform(df['Gender'])

# Pastikan daftar pekerjaan lengkap dengan label baru jika ada
occupation_labels = [
    'Software Engineer', 'Doctor', 'Sales Representative', 'Teacher', 'Nurse', 
    'Engineer', 'Accountant', 'Scientist', 'Lawyer', 'Manager', 'Salesperson', 'Others'
]
occupation_enc = LabelEncoder()
occupation_enc.fit(occupation_labels)  # Fit dengan semua kategori pekerjaan yang ada
df['Occupation_num'] = occupation_enc.transform(df['Occupation'])

bmi_enc = LabelEncoder()
df['BMI_num'] = bmi_enc.fit_transform(df['BMI Category'])

target_enc = LabelEncoder()
df['Target'] = target_enc.fit_transform(df['Sleep Disorder'])

print(f"✅ Label di target encoder: {list(target_enc.classes_)}")

# 4️⃣ Siapkan fitur & target
X = df[[ 
    'Gender_num', 'Age', 'Occupation_num', 'Sleep Duration', 'Quality of Sleep',
    'Physical Activity Level', 'Stress Level', 'BMI_num',
    'Heart Rate', 'Daily Steps', 'Systolic', 'Diastolic'
]]
y = df['Target']

# 5️⃣ Scale fitur numerik (tanpa gender & occupation)
scaler = MinMaxScaler()
X_scaled = X.copy()
cols_to_scale = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level',
                 'Stress Level', 'Heart Rate', 'Daily Steps', 'Systolic', 'Diastolic']
X_scaled[cols_to_scale] = scaler.fit_transform(X_scaled[cols_to_scale])

# 6️⃣ Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 7️⃣ Train model Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_acc = dt.score(X_test, y_test)
print(f"✅ Acc Decision Tree: {dt_acc:.4f}")

# Train model Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
rf_acc = rf.score(X_test, y_test)
print(f"✅ Acc Random Forest: {rf_acc:.4f}")

# Train model Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
lr_acc = lr.score(X_test, y_test)
print(f"✅ Acc Logistic Regression: {lr_acc:.4f}")

# 8️⃣ Simpan model & encoder
joblib.dump(dt, "best_model_decision_tree.pkl")
joblib.dump(rf, "random_forest_model.pkl")
joblib.dump(lr, "logistic_regression_model.pkl")
joblib.dump(gender_enc, "Gender_label_encoder.pkl")
joblib.dump(occupation_enc, "Occupation_label_encoder.pkl")
joblib.dump(bmi_enc, "BMI Category_label_encoder.pkl")
joblib.dump(target_enc, "target_label_encoder.pkl")
joblib.dump(scaler, "minmax_scaler_split.pkl")

print("✅ Semua model & encoder berhasil dilatih dan disimpan!") 