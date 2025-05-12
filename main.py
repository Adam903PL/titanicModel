import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('data.csv')

df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)

y = df['Survived']
X = df.drop(['Survived'], axis=1)

X = pd.get_dummies(X, drop_first=True)

X['Age'].fillna(X['Age'].mean(), inplace=True)

if 'Embarked' in X.columns:
    X['Embarked'].fillna(X['Embarked'].mode()[0], inplace=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f'Random Forest Accuracy: {acc_rf:.4f}')

lr = LogisticRegression(
    solver='saga',
    random_state=42,
    max_iter=1000,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=5,
    tol=1e-4
)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
acc_lr = accuracy_score(y_test, y_pred_lr)
print(f'Logistic Regression Accuracy: {acc_lr:.4f}')
