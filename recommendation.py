import pickle 
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE # If you choose to use SMOTE for class imbalance

# Load and preprocess your dataset
heart_data = pd.read_csv(r'D:\ML Ops\Heart fail prediction\heart.csv')
heart_data
# Replace this with your dataset loading and preprocessing code
data = pd.get_dummies(heart_data, columns=['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'], drop_first=True)
data
# Split data into features (X) and target (y)
X = data.drop("HeartDisease", axis=1)
y = data["HeartDisease"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature Selection
# Implement feature selection using RFE or feature importance scores
from sklearn.feature_selection import RFE
estimator = SVC(kernel="linear")
selector = RFE(estimator, n_features_to_select=10, step=1)
selector = selector.fit(X_train_scaled, y_train)
X_train_selected = selector.transform(X_train_scaled)

# Class Imbalance Handling
# Implement class imbalance handling using techniques like SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

# Ensemble Methods
# Implement ensemble methods like Random Forest or Gradient Boosting
ensemble_model = RandomForestClassifier()
ensemble_model.fit(X_train_scaled, y_train)

# Kernel Trick and Custom Kernels
# Implement SVM with different kernels and parameters
kernel_model = SVC(kernel='poly', C=1, gamma='scale')
kernel_model.fit(X_train_scaled, y_train)

# Regularization
# Implement SVM with different regularization parameters
regularization_model = SVC(C=0.1, kernel='linear', gamma='scale')
regularization_model.fit(X_train_scaled, y_train)

# After experimenting, choose the best model from the above strategies
best_model = SVC(random_state=42, C=1, kernel='rbf', gamma='scale')
best_model.fit(X_train_scaled, y_train)

pickle.dump(best_model,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))