import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

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

model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

model.fit(
    X_train_scaled, y_train,
    validation_split=0.1,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

y_pred_probs = model.predict(X_test_scaled).flatten()
y_pred = (y_pred_probs > 0.5).astype(int)
acc = accuracy_score(y_test, y_pred)
print(f'Keras Neural Network Accuracy: {acc:.4f}')
