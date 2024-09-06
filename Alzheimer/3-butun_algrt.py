from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Resimleri yükle ve ön işleme fonksiyonu
def load_and_preprocess_data(folder_path, label):
    data = []
    labels = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64, 64))
        data.append(img)
        labels.append(label)
    return data, labels

# Verileri yükleme ve etiketleri oluşturma
mild_demented_data, mild_demented_labels = load_and_preprocess_data('C:/Users/Casper/Desktop/Makine_Ogrenmesi/Alzheimer/Dataset/Mild_Demented', 0)
moderate_demented_data, moderate_demented_labels = load_and_preprocess_data('C:/Users/Casper/Desktop/Makine_Ogrenmesi/Alzheimer/Dataset/Moderate_Demented', 1)
non_demented_data, non_demented_labels = load_and_preprocess_data('C:/Users/Casper/Desktop/Makine_Ogrenmesi/Alzheimer/Dataset/Non_Demented', 2)
very_mild_demented_data, very_mild_demented_labels = load_and_preprocess_data('C:/Users/Casper/Desktop/Makine_Ogrenmesi/Alzheimer/Dataset/Very_Mild_Demented', 3)

# Verileri birleştir
data = np.array(mild_demented_data + moderate_demented_data + non_demented_data + very_mild_demented_data)
labels = np.array(mild_demented_labels + moderate_demented_labels + non_demented_labels + very_mild_demented_labels)

# Etiketleri one-hot encoding'e çevir
labels_one_hot = to_categorical(labels, num_classes=4)

# Veriyi eğitim ve test setlerine ayır
train_data, test_data, train_labels, test_labels = train_test_split(data, labels_one_hot, test_size=0.2, random_state=42)

# Piksel değerlerini 0 ile 1 arasında normalize et
train_data = train_data / 255.0
test_data = test_data / 255.0

# Veriyi yeniden şekillendirme
n_samples_train, height, width = train_data.shape
n_samples_test, _, _ = test_data.shape

train_data_flatten = train_data.reshape((n_samples_train, height * width))
test_data_flatten = test_data.reshape((n_samples_test, height * width))


# SVM modelini oluşturma ve eğitme
svm_model = SVC(kernel='linear', random_state=42)#linear 97 veriyor rbf de 73 e düştü.
svm_model.fit(train_data_flatten, np.argmax(train_labels, axis=1))
svm_predictions = svm_model.predict(test_data_flatten)

# Lojistik Regresyon modelini oluşturma ve eğitme
logistic_model = LogisticRegression(max_iter=1000, random_state=42)
logistic_model.fit(train_data_flatten, np.argmax(train_labels, axis=1))
logistic_predictions = logistic_model.predict(test_data_flatten)

# Rastgele Orman modelini oluşturma ve eğitme
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(train_data_flatten, np.argmax(train_labels, axis=1))
rf_predictions = rf_model.predict(test_data_flatten)


# Karar Ağacı modelini oluşturma ve eğitme
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(train_data_flatten, np.argmax(train_labels, axis=1))
dt_predictions = dt_model.predict(test_data_flatten)

# Model performanslarını değerlendirme
print("Support Vector Machine (SVM):")
print(confusion_matrix(np.argmax(test_labels, axis=1), svm_predictions))
print(classification_report(np.argmax(test_labels, axis=1), svm_predictions, digits=6))

print("\nLogistic Regression:")
print(confusion_matrix(np.argmax(test_labels, axis=1), logistic_predictions))
print(classification_report(np.argmax(test_labels, axis=1), logistic_predictions, digits=6))

print("\nRandom Forest:")
print(confusion_matrix(np.argmax(test_labels, axis=1), rf_predictions))
print(classification_report(np.argmax(test_labels, axis=1), rf_predictions, digits=6))

print("\nDecision Tree:")
print(confusion_matrix(np.argmax(test_labels, axis=1), dt_predictions))
print(classification_report(np.argmax(test_labels, axis=1), dt_predictions, digits=6))

# Support Vector Machine (SVM)
plt.figure(figsize=(8, 6))
svm_cm = confusion_matrix(np.argmax(test_labels, axis=1), svm_predictions)
sns.heatmap(svm_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented'], yticklabels=['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented'])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix - Support Vector Machine (SVM)')
plt.show()

# Logistic Regression
plt.figure(figsize=(8, 6))
logistic_cm = confusion_matrix(np.argmax(test_labels, axis=1), logistic_predictions)
sns.heatmap(logistic_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented'], yticklabels=['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented'])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix - Logistic Regression')
plt.show()

# Random Forest
plt.figure(figsize=(8, 6))
rf_cm = confusion_matrix(np.argmax(test_labels, axis=1), rf_predictions)
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented'], yticklabels=['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented'])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix - Random Forest')
plt.show()

# Decision Tree
plt.figure(figsize=(8, 6))
dt_cm = confusion_matrix(np.argmax(test_labels, axis=1), dt_predictions)
sns.heatmap(dt_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented'], yticklabels=['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented'])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix - Decision Tree')
plt.show()