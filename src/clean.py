import pandas as pd
import numpy as np

def load_data(path="data/raw/titanic.csv"):
    return pd.read_csv(path)

def clean_titanic(df):
    df = df.copy()  

# drop colomns with missing values and not usable
    df = df.drop(columns=["Cabin", "Ticket", "Name", "PassengerId"], errors="ignore")

#fill missing ages with a grouped (group by pclass, sex) median age
    df["Age"] = df.groupby(["Pclass", "Sex"])["Age"].transform(lambda x: x.fillna(x.median()))

#find the most common value and replace every blank with it.
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

#Add new colomns Family size, IsAlone, AgeGroup, FarePerPerson
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 12, 18, 35, 60, 100],
        labels=["Child", "Teen", "YoungAdult", "Adult", "Senior"],
    )
    df["FarePerPerson"] = df["Fare"] / df["FamilySize"]

#Converting text to numbers
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df = pd.get_dummies(df, columns=["Embarked", "AgeGroup"], drop_first=True)

    return df

#main function
if __name__ == "__main__":
    df = load_data()
    cleaned = clean_titanic(df)
    cleaned.to_csv("data/cleaned_titanic.csv", index=False)
    print(f"Cleaned shape: {cleaned.shape}")
    print(f"Remaining nulls:\n{cleaned.isnull().sum().sum()}")