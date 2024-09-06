import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
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

# Eğitim ve test verilerini düzleştirme
train_data_flat = train_data.reshape(train_data.shape[0], -1).astype('float32')
test_data_flat = test_data.reshape(test_data.shape[0], -1).astype('float32')

# Eğitim ve test etiketlerini düzeltme
train_labels_flat = np.argmax(train_labels, axis=1).astype('float32')
test_labels_flat = np.argmax(test_labels, axis=1).astype('float32')

# K-NN with GridSearchCV
param_grid = {'n_neighbors': [5], 'metric': ['euclidean', 'manhattan']}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, refit=True, verbose=3, cv=5)
grid.fit(train_data_flat, train_labels_flat)

# En iyi parametreleri ve skoru alın
print("Best Parameters:", grid.best_params_)
print("Best Score:", grid.best_score_)

# En iyi modeli seçin
knn_model = grid.best_estimator_

# 5 kat çapraz doğrulama için StratifiedKFold kullanarak skorları alın
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(knn_model, train_data_flat, train_labels_flat, cv=cv, scoring='accuracy')

# Skorları ekrana yazdırın
print("Accuracy Scores for Each Fold:", scores)

# Ortalama doğruluk skorunu yazdırın
print("Mean Accuracy:", np.mean(scores))

# Test verileri üzerinde tahmin yapma
labels_pred_flat = knn_model.predict(test_data_flat)

# Tahminleri yeniden one-hot encoding'e çevirme
labels_pred = to_categorical(labels_pred_flat, num_classes=4)

# Karmaşıklık Matrisi
conf_matrix_knn = confusion_matrix(np.argmax(test_labels, axis=1), labels_pred_flat)

# Karmaşıklık matrisini seaborn kullanarak görselleştir
class_names = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_knn, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix (K-NN)')
plt.xlabel('Predicted Value')
plt.ylabel('Actuel Value')
plt.show()

# Sınıflandırma Raporu
print(classification_report(np.argmax(test_labels, axis=1), labels_pred_flat, target_names=class_names, zero_division=1, digits=6))

from sklearn.metrics import f1_score, cohen_kappa_score

# Kappa katsayısı hesaplama
knn_kappa = cohen_kappa_score(np.argmax(test_labels, axis=1), labels_pred_flat)

# Modelin performansını yazdırma
print(f"KNN Model Kappa Score: {knn_kappa}")