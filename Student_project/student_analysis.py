import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

n = 120

# Create dataset
df = pd.DataFrame({
    "StudentID": np.arange(1, n+1),
    "Subject": np.random.choice(["Math", "Science", "English", "Computer"], n),
    "StudyHours": np.random.randint(1, 10, n),
    "Marks": np.random.randint(35, 100, n),
    "Attendance": np.random.randint(60, 100, n),
    "City": np.random.choice(["Chennai", "Coimbatore", "Madurai", "Salem"], n)
})

print("First 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

# ---------------------------
# DATA CLEANING
# ---------------------------
df = df.dropna()

# ---------------------------
# SUMMARY
# ---------------------------
print("\nSummary Statistics:")
print(df.describe())

# ---------------------------
# ANALYSIS 1 (Subject-wise Marks)
# ---------------------------
subject_marks = df.groupby("Subject")["Marks"].mean()

print("\nAverage Marks by Subject:")
print(subject_marks)

# ---------------------------
# ANALYSIS 2 (Study Hours vs Marks)
# ---------------------------
print("\nCorrelation:")
print(df[["StudyHours", "Marks"]].corr())

# ---------------------------
# VISUALIZATION (ONLY 2)
# ---------------------------

# 1️⃣ Bar Chart
plt.figure(figsize=(6,4))
subject_marks.plot(kind='bar')
plt.title("Average Marks by Subject")
plt.xlabel("Subject")
plt.ylabel("Marks")
plt.show()

# 2️⃣ Scatter Plot
plt.figure(figsize=(6,4))
plt.scatter(df["StudyHours"], df["Marks"])
plt.title("Study Hours vs Marks")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.show()