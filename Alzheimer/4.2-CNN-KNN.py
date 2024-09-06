import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix,roc_curve, auc,classification_report, confusion_matrix, accuracy_score


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

# Veriyi eğitim, doğrulama ve test setlerine ayır
train_data, test_data, train_labels, test_labels = train_test_split(data, labels_one_hot, test_size=0.2, random_state=42)
train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.25, random_state=42)

# Piksel değerlerini 0 ile 1 arasında normalize et
train_data = train_data / 255.0
val_data = val_data / 255.0
test_data = test_data / 255.0

# CNN modelini oluştur
input_layer = Input(shape=(64, 64, 1))
conv1 = Conv2D(32, (3, 3), activation='relu')(input_layer)
pool1 = MaxPooling2D((2, 2))(conv1)
conv2 = Conv2D(64, (3, 3), activation='relu')(pool1)
pool2 = MaxPooling2D((2, 2))(conv2)
flatten = Flatten()(pool2)
# CNN özellik çıkarma
cnn_features = Dense(128, activation='relu')(flatten)
cnn_features = Dense(4, activation='softmax')(cnn_features)

# CNN modelini ve özellik çıkarma katmanını birleştir
cnn_model = Model(inputs=input_layer, outputs=cnn_features)

# CNN modelini derleme
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping callback'i oluşturma
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Modeli eğitme
history = cnn_model.fit(train_data, train_labels, epochs=10, batch_size=32, validation_data=(val_data, val_labels), callbacks=[early_stopping])

# CNN modelinden özellik çıkarma katmanlarını al
feature_extractor = Model(inputs=cnn_model.input, outputs=cnn_model.layers[-2].output)

# CNN özellik çıkarma
cnn_features_train = feature_extractor.predict(train_data)
cnn_features_test = feature_extractor.predict(test_data)
# Doğrulama verilerinden de özellik çıkarma
validation_features = feature_extractor.predict(val_data)

# One-hot encoding'den sınıf endekslerine dönüştürme
train_labels_index = np.argmax(train_labels, axis=1)
test_labels_index = np.argmax(test_labels, axis=1)
validation_labels_index = np.argmax(val_labels, axis=1)


# K-NN with GridSearchCV
param_grid = {'n_neighbors': [5], 'metric': ['euclidean', 'manhattan']}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, refit=True, verbose=3, cv=5)
grid.fit(cnn_features_train, train_labels_index)

# En iyi parametreleri ve skoru alın
print("Best Parameters:", grid.best_params_)
print("Best Score:", grid.best_score_)

# K-NN sınıflandırıcı
KNN_classifier = grid.best_estimator_

# Test seti üzerinde tahmin yapma
KNN_pred_labels_test = KNN_classifier.predict(cnn_features_test)

# Performansı değerlendirme (test seti için)
accuracy_test = accuracy_score(test_labels_index, KNN_pred_labels_test)
print("KNN Model Accuracy (Test Set):", accuracy_test)

# Karmaşıklık Matrisi (test seti için)
conf_matrix_knn_test = confusion_matrix(test_labels_index, KNN_pred_labels_test)
class_names = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_knn_test, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix (KNN - Test Set)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Doğrulama seti üzerinde tahmin yapma
KNN_pred_labels_validation = KNN_classifier.predict(validation_features)

# Performansı değerlendirme (doğrulama seti için)
accuracy_validation = accuracy_score(validation_labels_index, KNN_pred_labels_validation)
print("KNN Model Accuracy (Validation Set):", accuracy_validation)

# Karmaşıklık Matrisi (doğrulama seti için)
conf_matrix_knn_validation = confusion_matrix(validation_labels_index, KNN_pred_labels_validation)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_knn_validation, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix (KNN - Validation Set)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Sınıflandırma Raporu
print("Sınıflandırma Raporu (Test Seti):")
print(classification_report(test_labels_index, KNN_pred_labels_test, target_names=class_names,digits=6))

# Doğruluk
accuracy = accuracy_score(validation_labels_index, KNN_pred_labels_validation)
print("Doğruluk:", accuracy)

# F1 skoru
f1 = f1_score(test_labels_index, KNN_pred_labels_test, average='weighted')
print("F1 Skoru:", f1)

# Kappa katsayısı
kappa = cohen_kappa_score(test_labels_index, KNN_pred_labels_test)
print("Kappa Katsayısı:", kappa)
