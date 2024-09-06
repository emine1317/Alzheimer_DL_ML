import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization
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

# Veriyi eğitim, test ve doğrulama setlerine ayır
train_data, temp_data, train_labels, temp_labels = train_test_split(data, labels_one_hot, test_size=0.2, random_state=42)
test_data, validation_data, test_labels, validation_labels = train_test_split(temp_data, temp_labels, test_size=0.5, random_state=42)

# Piksel değerlerini 0 ile 1 arasında normalize et
train_data = train_data / 255.0
test_data = test_data / 255.0
validation_data = validation_data / 255.0

# CNN modelini oluştur --->0.98593
cnn_model = Sequential()
cnn_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
cnn_model.add(BatchNormalization())
cnn_model.add(MaxPooling2D((2, 2)))
cnn_model.add(Conv2D(64, (3, 3), activation='relu'))
cnn_model.add(BatchNormalization())
cnn_model.add(MaxPooling2D((2, 2)))
cnn_model.add(Flatten())
cnn_model.add(Dense(128, activation='relu'))
cnn_model.add(Dense(4, activation='softmax'))  # 4 sınıf var

# Modeli derleme
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Veriyi yeniden şekillendirme
train_data_reshaped = train_data.reshape((train_data.shape[0], 64, 64, 1))
test_data_reshaped = test_data.reshape((test_data.shape[0], 64, 64, 1))
validation_data_reshaped = validation_data.reshape((validation_data.shape[0], 64, 64, 1))

# Early stopping callback'i oluşturma
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Modeli eğitme
history = cnn_model.fit(train_data_reshaped, train_labels, epochs=10, batch_size=32, validation_data=(validation_data_reshaped, validation_labels), callbacks=[early_stopping])

# CNN modelinden özellik çıkarma katmanlarını al
feature_extractor = Model(inputs=cnn_model.input, outputs=cnn_model.layers[-2].output)

# Eğitim ve test verilerinden özellik çıkarma
train_features = feature_extractor.predict(train_data_reshaped)
test_features = feature_extractor.predict(test_data_reshaped)

# Doğrulama verilerinden de özellik çıkarma
validation_features = feature_extractor.predict(validation_data_reshaped)

# One-hot encoding'den sınıf endekslerine dönüştürme
train_labels_index = np.argmax(train_labels, axis=1)
test_labels_index = np.argmax(test_labels, axis=1)
validation_labels_index = np.argmax(validation_labels, axis=1)

# SVM modelini oluştur
svm_model = SVC(kernel='sigmoid') #sigmoid %91

# SVM modelini eğit
svm_model.fit(train_features, train_labels_index)

# Test verilerini kullanarak modeli değerlendir
accuracy = svm_model.score(test_features, test_labels_index)
print("SVM Model Accuracy (Test Set):", accuracy)

# Doğrulama verilerini kullanarak modeli değerlendir
accuracy_validation = svm_model.score(validation_features, validation_labels_index)
print("SVM Model Accuracy (Validation Set):", accuracy_validation)

# Eğitim ve doğrulama doğruluklarını çizme
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy Values')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# Test verileri üzerinde tahmin yapma
svm_pred_labels = svm_model.predict(test_features)

# Performansı değerlendirme
accuracy = accuracy_score(test_labels_index, svm_pred_labels)
print("SVM Model Accuracy (Test Set):", accuracy)

# Doğrulama verileri üzerinde tahmin yapma
svm_pred_labels_validation = svm_model.predict(validation_features)

# Performansı değerlendirme (doğrulama seti için)
accuracy_validation = accuracy_score(validation_labels_index, svm_pred_labels_validation)
print("SVM Model Accuracy (Validation Set):", accuracy_validation)

# Karmaşıklık Matrisi
conf_matrix_svm = confusion_matrix(test_labels_index, svm_pred_labels)

# Karmaşıklık matrisini seaborn kullanarak görselleştir
class_names = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_svm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix  (CNN-SVM)')
plt.xlabel('Predicted Values')
plt.ylabel('Actuel Values')
plt.show()

# Sınıflandırma Raporu
print("Sınıflandırma Raporu (Test Seti):")
print(classification_report(test_labels_index, svm_pred_labels, target_names=class_names,digits=6))

# Sınıflandırma Raporu (doğrulama seti için)
print("Sınıflandırma Raporu (Validation Seti):")
print(classification_report(validation_labels_index, svm_pred_labels_validation, target_names=class_names))

# Doğruluk
accuracy = accuracy_score(test_labels_index, svm_pred_labels)
print("Doğruluk:", accuracy)

# F1 skoru
f1 = f1_score(test_labels_index, svm_pred_labels, average='weighted')
print("F1 Skoru:", f1)

# Kappa katsayısı
kappa = cohen_kappa_score(test_labels_index, svm_pred_labels)
print("Kappa Katsayısı:", kappa)
