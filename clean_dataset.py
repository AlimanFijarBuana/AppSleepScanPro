import pandas as pd

# 1️⃣ Load dataset asli dari Kaggle
df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")

# 2️⃣ Cek label awal
print("Label unik sebelum pembersihan:", df['Sleep Disorder'].unique())
print("Jumlah NaN di Sleep Disorder:", df['Sleep Disorder'].isna().sum())

# 3️⃣ Ganti NaN di Sleep Disorder dengan 'Good'
df['Sleep Disorder'] = df['Sleep Disorder'].fillna('Good')

# 4️⃣ Cek label setelah konversi
print("Label unik setelah pembersihan:", df['Sleep Disorder'].unique())
print("Jumlah label 'Good':", (df['Sleep Disorder'] == 'Good').sum())

# 5️⃣ Simpan ke file baru
df.to_csv("Sleep_dataset_cleaned.csv", index=False)
print("✅ Dataset bersih disimpan ke Sleep_dataset_cleaned.csv")
