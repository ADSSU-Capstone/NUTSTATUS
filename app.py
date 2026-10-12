import pandas as pd
import numpy as np
from scipy import stats
from io import StringIO

# Load the data (copy the table from your Excel file or use pd.read_excel)
# For demonstration, I'll assume you have the data in a DataFrame called df.
# If you have the file, use: df = pd.read_excel('NS-AA-B-0-J-1.xlsx', sheet_name='Sheet1')
# For now, I'll provide the code structure. You can replace the data loading with your actual file.

# After loading, clean the data:
# - Remove rows with empty Respondent ID or that are headers (GRADE 4, GRADE 5, GRADE 6)
# - Assign Grade Level based on the section
# - Convert numeric columns

# Here's the complete script:

import pandas as pd
import numpy as np
from scipy import stats

# Load data
df = pd.read_excel('NS-AA-B-0-J-1.xlsx', sheet_name='Sheet1')

# Clean: remove rows where Respondent ID is not numeric
df = df[pd.to_numeric(df['Respondent ID'], errors='coerce').notna()].copy()

# Assign grade level based on Respondent ID ranges (as per your data)
# Grade 4: IDs 1-44, Grade 5: 45-132, Grade 6: 133-214
df['Respondent ID'] = df['Respondent ID'].astype(int)
df['Grade Level'] = np.where(df['Respondent ID'] <= 44, 'Grade 4',
                    np.where(df['Respondent ID'] <= 132, 'Grade 5', 'Grade 6'))

# Convert numeric columns
numeric_cols = ['Height (m)', 'Weight (kg)', 'BMI', 'Filipino', 'English', 'Math', 'Science', 
                'A.P', 'EPP', 'GMRC', 'MAPEH']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Compute General Average (mean of 8 subjects, ignoring missing)
subject_cols = ['Filipino', 'English', 'Math', 'Science', 'A.P', 'EPP', 'GMRC', 'MAPEH']
df['General Average'] = df[subject_cols].mean(axis=1)

# 1. Demographic Profile
print("=== Demographic Profile ===")
demo = df['Grade Level'].value_counts().sort_index()
demo_perc = df['Grade Level'].value_counts(normalize=True).sort_index() * 100
for grade in demo.index:
    print(f"{grade}: f={demo[grade]}, %={demo_perc[grade]:.2f}")

# 2. Nutritional Status Overall
print("\n=== Nutritional Status Overall ===")
bmi_counts = df['BMI Category'].value_counts()
bmi_perc = df['BMI Category'].value_counts(normalize=True) * 100
for cat in bmi_counts.index:
    print(f"{cat}: f={bmi_counts[cat]}, %={bmi_perc[cat]:.2f}")

# 3. Nutritional Status by Grade Level
print("\n=== Nutritional Status by Grade Level ===")
crosstab = pd.crosstab(df['Grade Level'], df['BMI Category'])
print(crosstab)
print("\nPercentages:")
print(pd.crosstab(df['Grade Level'], df['BMI Category'], normalize='index') * 100)

# 4. Academic Achievements: General Average by Grade Level
print("\n=== General Average by Grade Level ===")
for grade in ['Grade 4', 'Grade 5', 'Grade 6']:
    subset = df[df['Grade Level'] == grade]['General Average'].dropna()
    mean = subset.mean()
    sd = subset.std()
    print(f"{grade}: Mean={mean:.2f}, SD={sd:.2f}, n={len(subset)}")

# Overall
overall = df['General Average'].dropna()
print(f"Overall: Mean={overall.mean():.2f}, SD={overall.std():.2f}, n={len(overall)}")

# 5. Academic Achievements by Subject Area
print("\n=== Subject Means and SD by Grade Level ===")
for grade in ['Grade 4', 'Grade 5', 'Grade 6']:
    print(f"\n{grade}:")
    subset = df[df['Grade Level'] == grade]
    for subj in subject_cols:
        mean = subset[subj].mean()
        sd = subset[subj].std()
        print(f"  {subj}: Mean={mean:.2f}, SD={sd:.2f}")

# 6. Correlation between BMI and General Average
print("\n=== Correlation: BMI vs General Average ===")
valid = df[['BMI', 'General Average']].dropna()
r, p = stats.pearsonr(valid['BMI'], valid['General Average'])
print(f"Pearson r = {r:.4f}, p-value = {p:.4f}, n = {len(valid)}")
if p < 0.05:
    print("Decision: Reject Ho - Significant relationship")
else:
    print("Decision: Accept Ho - No significant relationship")

# 7. ANOVA: BMI by Grade Level
print("\n=== ANOVA: BMI by Grade Level ===")
groups = [df[df['Grade Level'] == g]['BMI'].dropna() for g in ['Grade 4', 'Grade 5', 'Grade 6']]
f_val, p_val = stats.f_oneway(*groups)
print(f"F-value = {f_val:.4f}, p-value = {p_val:.4f}")
if p_val < 0.05:
    print("Decision: Reject Ho - Significant difference")
else:
    print("Decision: Accept Ho - No significant difference")
