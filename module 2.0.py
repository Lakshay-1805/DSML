
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, chi2_contingency, f_oneway


# -------------------- LOAD DATA --------------------

ab = pd.read_csv(r"C:\Users\laksh\Downloads\ab_data.csv")
countries = pd.read_csv(r"C:\Users\laksh\Downloads\countries.csv")

print("Original data:", ab.shape)


# -------------------- DATA CLEANING --------------------

# Remove group/landing-page mismatches
ab = ab[
    ~(
        ((ab["group"] == "control") & (ab["landing_page"] == "old_page")) |
        ((ab["group"] == "treatment") & (ab["landing_page"] == "new_page"))
    )
]

# Remove duplicate users
ab = ab.drop_duplicates("user_id")

print("Cleaned data:", ab.shape)


# -------------------- MERGE COUNTRY DATA --------------------

countries = countries.drop_duplicates("user_id")

ab = pd.merge(
    ab,
    countries,
    on="user_id",
    how="left"
)

print("Merged data:", ab.shape)

print("\nFirst 5 rows:")
print(ab.head())


# -------------------- BASIC INFORMATION --------------------

print("\nData Information:")
print(ab.info())

print("\nMissing Values:")
print(ab.isnull().sum())

print("\nGroup Counts:")
print(ab["group"].value_counts())


# -------------------- CONVERSION RATE --------------------

conversion_rate = ab.groupby("group")["converted"].mean()

print("\nConversion Rate:")
print(conversion_rate)


# -------------------- A/B TEST --------------------

control = ab[ab["group"] == "control"]["converted"]
treatment = ab[ab["group"] == "treatment"]["converted"]

t_stat, p_value = ttest_ind(control, treatment)

print("\nA/B Test Results:")
print("T-statistic:", t_stat)
print("P-value:", p_value)

if p_value < 0.05:
    print("Result: Statistically significant difference.")
else:
    print("Result: No statistically significant difference.")


# -------------------- CHI-SQUARE TEST --------------------

contingency_table = pd.crosstab(
    ab["group"],
    ab["converted"]
)

chi2, p, dof, expected = chi2_contingency(contingency_table)

print("\nChi-Square Test:")
print("Chi-square:", chi2)
print("P-value:", p)
print("Degrees of freedom:", dof)


# -------------------- ANOVA --------------------

control_data = ab[ab["group"] == "control"]["converted"]
treatment_data = ab[ab["group"] == "treatment"]["converted"]

f_stat, anova_p = f_oneway(control_data, treatment_data)

print("\nANOVA Test:")
print("F-statistic:", f_stat)
print("P-value:", anova_p)


# -------------------- VISUALIZATION --------------------

plt.figure(figsize=(7, 5))

sns.barplot(
    data=ab,
    x="group",
    y="converted"
)

plt.title("Conversion Rate by Group")
plt.xlabel("Group")
plt.ylabel("Conversion Rate")

plt.show()


# -------------------- COUNTRY ANALYSIS --------------------

if "country" in ab.columns:

    country_conversion = (
        ab.groupby("country")["converted"]
        .mean()
        .sort_values(ascending=False)
    )

    print("\nConversion Rate by Country:")
    print(country_conversion)

    plt.figure(figsize=(10, 6))

    country_conversion.plot(kind="bar")

    plt.title("Conversion Rate by Country")
    plt.xlabel("Country")
    plt.ylabel("Conversion Rate")

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
