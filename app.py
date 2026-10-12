import pandas as pd
import numpy as np

# Load data
df = pd.read_excel('NS-AA-B-0-J-1.xlsx', sheet_name='Sheet1')

# Clean: remove non-numeric Respondent IDs (headers)
df = df[pd.to_numeric(df['Respondent ID'], errors='coerce').notna()].copy()
df['Respondent ID'] = df['Respondent ID'].astype(int)

# Assign grade level
df['Grade Level'] = np.where(df['Respondent ID'] <= 44, 'Grade 4',
                    np.where(df['Respondent ID'] <= 132, 'Grade 5', 'Grade 6'))

# Convert numeric
numeric_cols = ['Height (m)', 'Weight (kg)', 'BMI', 'Filipino', 'English', 'Math', 
                'Science', 'A.P', 'EPP', 'GMRC', 'MAPEH']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# General Average
subject_cols = ['Filipino', 'English', 'Math', 'Science', 'A.P', 'EPP', 'GMRC', 'MAPEH']
df['General Average'] = df[subject_cols].mean(axis=1)

# --- 1. DEMOGRAPHIC ---
print("=== 1. DEMOGRAPHIC PROFILE ===")
print(df['Grade Level'].value_counts().sort_index())

# --- 2. NUTRITIONAL STATUS OVERALL ---
print("\n=== 2. NUTRITIONAL STATUS (OVERALL) ===")
print(df['BMI Category'].value_counts())
print("\nPercentages:")
print((df['BMI Category'].value_counts(normalize=True) * 100).round(2))

# --- 3. NUTRITIONAL STATUS BY GRADE ---
print("\n=== 3. NUTRITIONAL STATUS BY GRADE ===")
ct = pd.crosstab(df['Grade Level'], df['BMI Category'])
print(ct)
print("\nRow %:")
print((pd.crosstab(df['Grade Level'], df['BMI Category'], normalize='index') * 100).round(2))

# --- 4. GENERAL AVERAGE BY GRADE ---
print("\n=== 4. GENERAL AVERAGE BY GRADE LEVEL ===")
for g in ['Grade 4', 'Grade 5', 'Grade 6']:
    s = df[df['Grade Level'] == g]['General Average'].dropna()
    print(f"{g}: Mean={s.mean():.2f}, SD={s.std():.2f}, n={len(s)}")
overall = df['General Average'].dropna()
print(f"Overall: Mean={overall.mean():.2f}, SD={overall.std():.2f}, n={len(overall)}")

# --- 5. SUBJECT MEANS ---
print("\n=== 5. SUBJECT MEANS AND SD (ALL GRADES COMBINED) ===")
for subj in subject_cols:
    s = df[subj].dropna()
    print(f"{subj}: Mean={s.mean():.2f}, SD={s.std():.2f}")

# --- 6. PEARSON r (Manual, no scipy) ---
print("\n=== 6. CORRELATION: BMI vs GENERAL AVERAGE ===")
valid = df[['BMI', 'General Average']].dropna()
x = valid['BMI'].values
y = valid['General Average'].values
n = len(x)
r = (n*np.sum(x*y) - np.sum(x)*np.sum(y)) / np.sqrt((n*np.sum(x**2) - np.sum(x)**2) * (n*np.sum(y**2) - np.sum(y)**2))
# p-value using t-distribution approximation
t_stat = r * np.sqrt((n-2)/(1-r**2))
# Two-tailed p-value using normal approximation (works for n>30)
from math import erfc, sqrt
p_val = erfc(abs(t_stat)/sqrt(2))
print(f"Pearson r = {r:.4f}, t = {t_stat:.4f}, p-value ≈ {p_val:.4f}, n = {n}")
print("Decision:", "Reject Ho (Significant)" if p_val < 0.05 else "Accept Ho (Not Significant)")

# --- 7. ANOVA (Manual) ---
print("\n=== 7. ANOVA: BMI by GRADE LEVEL ===")
groups = [df[df['Grade Level'] == g]['BMI'].dropna().values for g in ['Grade 4', 'Grade 5', 'Grade 6']]
k = len(groups)
N = sum(len(g) for g in groups)
grand_mean = np.concatenate(groups).mean()
# Between-group sum of squares
SSB = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
# Within-group sum of squares
SSW = sum(np.sum((g - g.mean())**2) for g in groups)
df_b = k - 1
df_w = N - k
MSB = SSB / df_b
MSW = SSW / df_w
F = MSB / MSW
# p-value (very rough approximation using F distribution)
from math import exp
# Use scipy if available:
try:
    from scipy import stats
    p_val = 1 - stats.f.cdf(F, df_b, df_w)
except ImportError:
    p_val = float('nan')
print(f"SSB={SSB:.2f}, df_b={df_b}, MSB={MSB:.2f}")
print(f"SSW={SSW:.2f}, df_w={df_w}, MSW={MSW:.2f}")
print(f"F = {F:.4f}, p ≈ {p_val:.4f}")
print("Decision:", "Reject Ho (Significant)" if p_val < 0.05 else "Accept Ho (Not Significant)")

# --- 8. GRADE-LEVEL MEANS AND SDs FOR SUBJECTS ---
print("\n=== 8. SUBJECT MEANS PER GRADE ===")
for g in ['Grade 4', 'Grade 5', 'Grade 6']:
    print(f"\n--- {g} ---")
    sub = df[df['Grade Level'] == g]
    for s in subject_cols:
        print(f"  {s}: Mean={sub[s].mean():.2f}, SD={sub[s].std():.2f}, n={sub[s].count()}")
