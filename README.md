# Simple-Linear-Regression
---

```markdown
# 💼 Salary Prediction Based on Experience

A beginner-friendly Machine Learning project using **Simple Linear Regression** to predict salaries based on years of experience. This project is ideal for students and newcomers to ML who want to understand regression, visualization, and model evaluation in Python using `Scikit-Learn`.

---

## 📌 Project Objective

Build a linear regression model to predict a person's **salary** based on their **years of experience**.

---

## 🧠 What You'll Learn

✅ Understanding how linear regression works  
✅ Visualizing data relationships using `Matplotlib`  
✅ Splitting data into training and testing sets  
✅ Training and evaluating a regression model  
✅ Calculating metrics like MSE and R² Score  
✅ Making predictions for new values  
✅ Interpreting regression coefficients

---

## 📊 Dataset Overview

- **Source**: [Kaggle - Salary Data](https://www.kaggle.com/datasets/karthickveerakumar/salary-data-simple-linear-regression) 
- **Filename**: `Salary_Data.csv`
- **Columns**:
  - `YearsExperience`: Independent variable
  - `Salary`: Dependent variable

💡 It’s a small dataset, perfect for simple modeling and fast experimentation.

---

## 🛠️ Tools & Libraries

| Tool | Use |
|------|-----|
| `Python 3.x` | Programming Language |
| `Pandas` | Data Handling |
| `NumPy` | Numerical Operations |
| `Matplotlib` | Visualization |
| `Scikit-Learn` | Machine Learning (Linear Regression) |

📦 Install all dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn
```

---

## 📂 Project Structure

```
Salary-Prediction/
│
├── Salary_Data.csv
├── salary_prediction.py  # or .ipynb if using Jupyter
├── README.md
└── requirements.txt      # optional
```

---

## 🚀 How It Works

### 1️⃣ Load the Dataset  
Load the CSV file using pandas and inspect the data.

### 2️⃣ Visualize the Data  
Plot `YearsExperience` vs `Salary` to see a linear trend.

### 3️⃣ Prepare the Data  
Split the data into training and testing sets.

### 4️⃣ Train the Model  
Use `LinearRegression()` to fit a model.

### 5️⃣ Evaluate Performance  
Check how well the model performs using R² score and MSE.

### 6️⃣ Predict New Values  
Input a custom experience (e.g., 5 years) and predict salary.

---

## 📈 Sample Output

| YearsExperience | Actual Salary | Predicted Salary |
|-----------------|----------------|------------------|
| 1.5             | $39,000        | $40,105          |
| 3.2             | $57,000        | $54,130          |
| 7.1             | $90,000        | $91,680          |

---

## 📉 Regression Line Visualization

![Salary vs Experience](https://upload.wikimedia.org/wikipedia/commons/3/3a/Linear_regression.svg)

*Sample illustration – your plot will be generated via Matplotlib.*

---

## 📌 Model Formula

The Linear Regression model learns the equation:

\[
\text{Salary} = \beta_0 + \beta_1 \cdot \text{YearsExperience}
\]

Where:
- \( \beta_0 \): Intercept
- \( \beta_1 \): Coefficient (slope of the line)

---

## 🧪 Model Evaluation

| Metric              | Meaning                             | Ideal Value |
|---------------------|-------------------------------------|-------------|
| Mean Squared Error  | Average of squared prediction error | Lower = Better |
| R² Score            | How well line fits the data         | Closer to 1 |

---

## 🛠 Sample Code

```python
from sklearn.linear_model import LinearRegression
import pandas as pd

df = pd.read_csv("Salary_Data.csv")
X = df[['YearsExperience']]
y = df['Salary']

model = LinearRegression()
model.fit(X, y)

experience = pd.DataFrame({'YearsExperience': [5]})
salary = model.predict(experience)

print(f"Predicted salary for 5 years: ${salary[0]:.2f}")
```

---

## 🧩 Common Warning & Fix

**Warning**:  
```bash
UserWarning: X does not have valid feature names, but LinearRegression was fitted with feature names
```

✅ **Fix**: Use a **DataFrame**, not a plain NumPy array when predicting:

```python
# Correct
model.predict(pd.DataFrame({'YearsExperience': [5]}))
```

---

## ✅ Future Enhancements

- Add multiple features (e.g., education, industry) — switch to **Multiple Linear Regression**
- Build a **Streamlit** or **Flask** web app
- Explore **polynomial regression** for nonlinear relationships

---

## 👤 Author

**Kalicharan**  
🎓 M.Tech in Computer Science & Data Processing  
🌱 Machine Learning | AI | Deep Learning  
📫 [LinkedIn](https://www.linkedin.com/feed/) | [GitHub](https://github.com/Kalicharan4104)

---

## 📄 License

This project is open to everyone, allowing access and usage for learning purposes-feel free to use, modify, and share!

---

⭐ If you found this helpful, consider starring the repo on GitHub!
```

---


Happy coding! 🚀
