# ============================================================
# NUTRITIONAL STATUS & ACADEMIC ACHIEVEMENTS — FULL ANALYSIS
# Run in Google Colab. Upload NS-AA-B-0-J-1.xlsx first.
# ============================================================
!pip install openpyxl -q

import pandas as pd
import numpy as np

# --- LOAD ---
df = pd.read_excel('NS-AA-B-0-J-1.xlsx', sheet_name='Sheet1')

# --- CLEAN ---
df = df[pd.to_numeric(df['Respondent ID'], errors='coerce').notna()].copy()
df['Respondent ID'] = df['Respondent ID'].astype(int)
df['Grade Level'] = np.where(df['Respondent ID'] <= 44, 'Grade 4',
                    np.where(df['Respondent ID'] <= 132, 'Grade 5', 'Grade 6'))

numeric_cols = ['Height (m)', 'Weight (kg)', 'BMI', 'Filipino', 'English', 'Math',
                'Science', 'A.P', 'EPP', 'GMRC', 'MAPEH']
for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

subject_cols = ['Filipino', 'English', 'Math', 'Science', 'A.P', 'EPP', 'GMRC', 'MAPEH']
df['General Average'] = df[subject_cols].mean(axis=1)

# ============================================================
print("=" * 60)
print("TABLE 1 — DEMOGRAPHIC PROFILE")
print("=" * 60)
demo = df['Grade Level'].value_counts().sort_index()
for g in demo.index:
    print(f"{g}: f={demo[g]}, %={demo[g]/214*100:.2f}")

# ============================================================
print("\n" + "=" * 60)
print("TABLE 2 — NUTRITIONAL STATUS OVERALL")
print("=" * 60)
bmi = df['BMI Category'].value_counts()
for c in bmi.index:
    print(f"{c}: f={bmi[c]}, %={bmi[c]/214*100:.2f}")

# ============================================================
print("\n" + "=" * 60)
print("TABLE 3 — NUTRITIONAL STATUS BY GRADE LEVEL")
print("=" * 60)
ct = pd.crosstab(df['Grade Level'], df['BMI Category'])
print(ct)
print("\nRow %:")
print((pd.crosstab(df['Grade Level'], df['BMI Category'], normalize='index') * 100).round(2))

# ============================================================
print("\n" + "=" * 60)
print("TABLE 4 — GENERAL AVERAGE BY GRADE LEVEL")
print("=" * 60)
for g in ['Grade 4', 'Grade 5', 'Grade 6']:
    s = df[df['Grade Level'] == g]['General Average'].dropna()
    print(f"{g}: Mean={s.mean():.2f}, SD={s.std():.2f}, n={len(s)}")
ov = df['General Average'].dropna()
print(f"OVERALL: Mean={ov.mean():.2f}, SD={ov.std():.2f}, n={len(ov)}")

# ============================================================
print("\n" + "=" * 60)
print("TABLE 5 — SUBJECT AREA MEANS AND SD (ALL GRADES)")
print("=" * 60)
for s in subject_cols:
    col = df[s].dropna()
    print(f"{s}: Mean={col.mean():.2f}, SD={col.std():.2f}")

# ============================================================
print("\n" + "=" * 60)
print("TABLE 6 — PEARSON r: BMI vs GENERAL AVERAGE")
print("=" * 60)
valid = df[['BMI', 'General Average']].dropna()
x, y = valid['BMI'].values, valid['General Average'].values
n = len(x)
r = (n*np.sum(x*y) - np.sum(x)*np.sum(y)) / np.sqrt(
    (n*np.sum(x**2) - np.sum(x)**2) * (n*np.sum(y**2) - np.sum(y)**2))
# p-value using t-dist (approx)
from scipy import stats
t_stat = r * np.sqrt((n-2)/(1-r**2))
p_val = 2 * (1 - stats.t.cdf(abs(t_stat), n-2))
print(f"Pearson r = {r:.4f}")
print(f"t = {t_stat:.4f}, df = {n-2}")
print(f"p-value = {p_val:.4f}")
print(f"n = {n}")
print("Decision:", "Reject Ho (SIGNIFICANT)" if p_val < 0.05 else "Accept Ho (NOT significant)")

# ============================================================
print("\n" + "=" * 60)
print("TABLE 7 — ANOVA: BMI by GRADE LEVEL")
print("=" * 60)
groups = [df[df['Grade Level'] == g]['BMI'].dropna().values for g in ['Grade 4','Grade 5','Grade 6']]
f_val, p_val = stats.f_oneway(*groups)
k = len(groups)
N = sum(len(g) for g in groups)
gm = np.concatenate(groups).mean()
SSB = sum(len(g)*(g.mean()-gm)**2 for g in groups)
SSW = sum(np.sum((g-g.mean())**2) for g in groups)
print(f"SSB = {SSB:.4f}, df_b = {k-1}, MSB = {SSB/(k-1):.4f}")
print(f"SSW = {SSW:.4f}, df_w = {N-k}, MSW = {SSW/(N-k):.4f}")
print(f"F = {f_val:.4f}, p = {p_val:.4f}")
print("Decision:", "Reject Ho (SIGNIFICANT)" if p_val < 0.05 else "Accept Ho (NOT significant)")