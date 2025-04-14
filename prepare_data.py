import pandas as pd
import os

print(os.listdir('data'))
# Load symptom-disease dataset
df_disease_raw = pd.read_csv('data/disease_dataset.csv')
print("Columns in df_disease:", df_disease_raw.columns.tolist())

# Melt the wide symptom columns into a single 'Symptom' column
symptom_columns = [col for col in df_disease_raw.columns if col.startswith('Symptom')]
df_disease_melted = df_disease_raw.melt(id_vars=['Disease'], value_vars=symptom_columns,
                                        var_name='SymptomIndex', value_name='Symptom')

# Drop missing and clean
df_disease_cleaned = df_disease_melted.dropna()
df_disease_cleaned['Symptom'] = df_disease_cleaned['Symptom'].str.lower().str.strip()
df_disease_cleaned['Disease'] = df_disease_cleaned['Disease'].str.lower().str.strip()

# Save
df_disease_cleaned.to_csv('data/cleaned_symptom_disease.csv', index=False)
print("✅ Saved cleaned_symptom_disease.csv")

# Load and clean symptom severity data
df_severity = pd.read_csv("data/symptom_severity.csv")
df_severity.columns = ['Symptom', 'Severity']
df_severity['Symptom'] = df_severity['Symptom'].str.lower().str.strip()


# Map numeric severity to categories
def map_severity(level):
    if level <= 2:
        return 'mild'
    elif level == 3:
        return 'moderate'
    else:
        return 'severe'

df_severity['SeverityLevel'] = df_severity['Severity'].apply(map_severity)

# Save preprocessed versions
df_severity[['Symptom', 'SeverityLevel']].to_csv('data/cleaned_symptom_severity.csv', index=False)



df_disease = pd.read_csv('data/cleaned_symptom_disease.csv')
print(df_disease.head(10))
print(df_disease.shape)
print(df_disease['Symptom'].nunique(), "unique symptoms")
print(df_disease['Disease'].nunique(), "unique diseases")
df_severity = pd.read_csv('data/cleaned_symptom_severity.csv')
print(df_severity.head(10))
print(df_severity['SeverityLevel'].value_counts())
print(df_severity.shape)
