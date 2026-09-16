# Titanic — Exploratory Data Analysis

Exploratory analysis of the Titanic passenger dataset, with a reusable
cleaning pipeline and charts highlighting the main survival drivers.

## Dataset

Source: [Kaggle Titanic](https://www.kaggle.com/c/titanic/data)
Mirror: [GitHub raw CSV](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)

The raw file is expected at `data/raw/titanic.csv`. To fetch it:

```bash
curl -o data/raw/titanic.csv https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv

# Project Structure

titanic-eda/
├── data/
│   ├── raw/titanic.csv
│   └── cleaned_titanic.csv
├── src/
│   └── clean.py
├── notebooks/
│   └── titanic_eda.ipynb
├── charts/
├── requirements.txt
├── README.md
└── .gitignore


# Setup
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows
# source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt

# requirements.txt:
pandas>=2.2
numpy>=1.26
matplotlib>=3.8
seaborn>=0.13
notebook>=7.0

# Cleaning Pipeline
src/clean.py performs:
-Drop Cabin (>77% missing), Ticket, Name, PassengerId.
-Impute Age with median grouped by Pclass + Sex.
-Impute Embarked with mode (S).
-Feature engineering: FamilySize, IsAlone, AgeGroup, FarePerPerson.
-Encode Sex (0/1); one-hot encode Embarked and AgeGroup.

Run it:
-bash
-python src/clean.py

Output: data/cleaned_titanic.csv — 891 rows × 16 columns, no missing values.

#Notebook
Open notebooks/titanic_eda.ipynb in Jupyter or VS Code. It walks through loading, cleaning, visualizing, and documenting observations.

#Visualizations
Survival by class & gender- charts/survival_by_class_gender.png
Age distribution- charts/age_distribution.png
Correlation heatmap- charts/correlation_heatmap.png
Survival by family size- charts/survival_by_family.png

#Key Findings
-Gender and class dominate. Female ~74% vs male ~19%; 1st ~63% vs 3rd ~24%.
-Age is U-shaped. Children survived more; working-age adults died most.
-Family size 2–4 was optimal. Solo and large families fared worse.
-Top predictors: Sex, Pclass, FarePerPerson, FamilySize.

#Conclusion
-Gender and passenger class are the dominant survival drivers. Age shows a
-U-shaped effect — children benefited most, working-age adults suffered the
-highest losses. Family size of 2–4 maximizes survival. The cleaned dataset
-(no nulls, engineered features) is ready for logistic regression or
tree-based classifiers.

#Reproducing
pip install -r requirements.txt
curl -o data/raw/titanic.csv https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
python src/clean.py
jupyter notebook notebooks/titanic_eda.ipynb

