import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Titanic EDA", layout="wide")
st.title("🚢 Titanic — Exploratory Data Analysis")

@st.cache_data
def load_data():
    return pd.read_csv("data/cleaned_titanic.csv")

df = load_data()

st.markdown("### Key Findings")
st.markdown("""
- **Gender & class dominate survival**: females ~74%, males ~19%; 1st class ~3× 3rd class survival.
- **Age shows a U-shaped pattern**: children and young adults had higher survival odds.
- **Optimal family size = 2–4**: solo travelers and large families (5+) fared worst.
""")

# Chart 1: Survival by Class & Gender
col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x="Pclass", y="Survived", data=df, palette="Blues_d", ax=ax)
    ax.set_title("Survival Rate by Class")
    st.pyplot(fig)
with col2:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x="Sex", y="Survived", data=df, palette="Set2", ax=ax)
    ax.set_xticklabels(["Male", "Female"])
    ax.set_title("Survival Rate by Gender")
    st.pyplot(fig)

# Chart 2: Age Distribution
st.markdown("### Age Distribution by Survival")
fig, ax = plt.subplots(figsize=(10, 4))
sns.histplot(data=df, x="Age", hue="Survived", kde=True, bins=30,
             palette={0: "#e74c3c", 1: "#2ecc71"}, ax=ax)
st.pyplot(fig)

# Chart 3: Correlation Heatmap
st.markdown("### Correlation Heatmap")
num_cols = ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch",
            "Fare", "FamilySize", "IsAlone", "FarePerPerson"]
corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, square=True, linewidths=0.5, ax=ax)
st.pyplot(fig)

st.markdown("---")
st.caption("Built with Streamlit · Data: Kaggle Titanic")