import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats

st.title("Nutritional Status & Academic Achievements — Analysis")

# --- LOAD ---
df = pd.read_excel('NS-AA-B-0-J-1.xlsx', sheet_name='Sheet1')

# --- CLEAN COLUMN NAMES (strip spaces) ---
df.columns = df.columns.astype(str).str.strip()

# --- DEBUG: show actual column names ---
st.subheader("Detected Columns")
st.write(list(df.columns))

# --- Remove non-numeric Respondent ID rows (headers like "GRADE 4") ---
df = df[pd.to_numeric(df['Respondent ID'], errors='coerce').notna()].copy()
df['Respondent ID'] = df['Respondent ID'].astype(int)

# --- Assign Grade Level ---
df['Grade Level'] = np.where(df['Respondent ID'] <= 44, 'Grade 4',
                    np.where(df['Respondent ID'] <= 132, 'Grade 5', 'Grade 6'))

# --- Flexible column matching ---
def find_col(df, *candidates):
    for c in candidates:
        if c in df.columns:
            return c
    # try case-insensitive
    lower_map = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    raise KeyError(f"None of {candidates} found in {list(df.columns)}")

height_col = find_col(df, 'Height (m)', 'Height', 'height')
weight_col = find_col(df, 'Weight (kg)', 'Weight', 'weight')
bmi_col = find_col(df, 'BMI', 'Bmi', 'bmi')

subject_map = {
    'Filipino': find_col(df, 'Filipino'),
    'English': find_col(df, 'English'),
    'Math': find_col(df, 'Math', 'Mathematics'),
    'Science': find_col(df, 'Science'),
    'A.P': find_col(df, 'A.P', 'AP', 'Araling Panlipunan'),
    'EPP': find_col(df, 'EPP', 'EPP/TLE', 'TLE'),
    'GMRC': find_col(df, 'GMRC'),
    'MAPEH': find_col(df, 'MAPEH'),
}

# --- Convert numeric ---
for col in [height_col, weight_col, bmi_col] + list(subject_map.values()):
    df[col] = pd.to_numeric(df[col], errors='coerce')

# --- General Average ---
df['General Average'] = df[list(subject_map.values())].mean(axis=1)
# --- FIX DATA ENTRY ERROR: Respondent 55 ---
df.loc[df['Respondent ID'] == 55, 'Weight (kg)'] = 28.5
df.loc[df['Respondent ID'] == 55, 'BMI'] = round(28.5 / (1.26 ** 2), 1)
df.loc[df['Respondent ID'] == 55, 'BMI Category'] = 'Normal'
# ============ TABLE 1 ============
st.header("Table 1 — Demographic Profile")
demo = df['Grade Level'].value_counts().sort_index()
for g in demo.index:
    st.write(f"{g}: f={demo[g]}, %={demo[g]/214*100:.2f}")

# ============ TABLE 2 ============
st.header("Table 2 — Nutritional Status Overall")
bmi_cat_col = find_col(df, 'BMI Category', 'BMI Cat', 'Category')
bmi = df[bmi_cat_col].value_counts()
for c in bmi.index:
    st.write(f"{c}: f={bmi[c]}, %={bmi[c]/214*100:.2f}")

# ============ TABLE 3 ============
st.header("Table 3 — Nutritional Status by Grade Level")
ct = pd.crosstab(df['Grade Level'], df[bmi_cat_col])
st.dataframe(ct)
st.write("Row %:")
st.dataframe((pd.crosstab(df['Grade Level'], df[bmi_cat_col], normalize='index') * 100).round(2))

# ============ TABLE 4 ============
st.header("Table 4 — General Average by Grade Level")
for g in ['Grade 4', 'Grade 5', 'Grade 6']:
    s = df[df['Grade Level'] == g]['General Average'].dropna()
    st.write(f"{g}: Mean={s.mean():.2f}, SD={s.std():.2f}, n={len(s)}")
ov = df['General Average'].dropna()
st.write(f"OVERALL: Mean={ov.mean():.2f}, SD={ov.std():.2f}, n={len(ov)}")

# ============ TABLE 5 ============
st.header("Table 5 — Subject Area Means and SD")
for label, col in subject_map.items():
    c = df[col].dropna()
    st.write(f"{label}: Mean={c.mean():.2f}, SD={c.std():.2f}")

# ============ TABLE 6 ============
st.header("Table 6 — Pearson r: BMI vs General Average")
valid = df[[bmi_col, 'General Average']].dropna()
x, y = valid[bmi_col].values, valid['General Average'].values
n = len(x)
r = (n*np.sum(x*y) - np.sum(x)*np.sum(y)) / np.sqrt(
    (n*np.sum(x**2) - np.sum(x)**2) * (n*np.sum(y**2) - np.sum(y)**2))
t_stat = r * np.sqrt((n-2)/(1-r**2))
p_val = 2 * (1 - stats.t.cdf(abs(t_stat), n-2))
st.write(f"Pearson r = {r:.4f}")
st.write(f"t = {t_stat:.4f}, df = {n-2}")
st.write(f"p-value = {p_val:.4f}")
st.write(f"n = {n}")
st.write("Decision: " + ("Reject Ho (SIGNIFICANT)" if p_val < 0.05 else "Accept Ho (NOT significant)"))

# ============ TABLE 7 ============
st.header("Table 7 — ANOVA: BMI by Grade Level")
groups = [df[df['Grade Level'] == g][bmi_col].dropna().values for g in ['Grade 4','Grade 5','Grade 6']]
f_val, p_val = stats.f_oneway(*groups)
k = len(groups)
N = sum(len(g) for g in groups)
gm = np.concatenate(groups).mean()
SSB = sum(len(g)*(g.mean()-gm)**2 for g in groups)
SSW = sum(np.sum((g-g.mean())**2) for g in groups)
st.write(f"SSB = {SSB:.4f}, df_b = {k-1}, MSB = {SSB/(k-1):.4f}")
st.write(f"SSW = {SSW:.4f}, df_w = {N-k}, MSW = {SSW/(N-k):.4f}")
st.write(f"F = {f_val:.4f}, p = {p_val:.4f}")
st.write("Decision: " + ("Reject Ho (SIGNIFICANT)" if p_val < 0.05 else "Accept Ho (NOT significant)"))
