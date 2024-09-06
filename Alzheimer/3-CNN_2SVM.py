import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
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

# CNN modelini ve özellik çıkarma katmanını birleştir
cnn_model = Sequential()
cnn_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
cnn_model.add(MaxPooling2D((2, 2)))
cnn_model.add(Conv2D(64, (3, 3), activation='relu'))
cnn_model.add(MaxPooling2D((2, 2)))
cnn_model.add(Flatten())
cnn_model.add(Dense(128, activation='relu'))
cnn_model.add(Dense(4, activation='softmax'))  # 4 sınıf var

# Modeli derleme
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Modeli eğitim için ayarla
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# CNN modelini eğitme
cnn_model.fit(train_data, train_labels, epochs=10, batch_size=32, validation_data=(test_data, test_labels), callbacks=[early_stopping])

# CNN özellik çıkarma
cnn_features_train = cnn_model.predict(train_data)
cnn_features_test = cnn_model.predict(test_data)

# SVM sınıflandırıcı
svm_classifier = SVC()

# SVM sınıflandırıcıyı eğitme
svm_classifier.fit(cnn_features_train, np.argmax(train_labels, axis=1))

# Test seti üzerinde tahmin yapma
svm_pred_labels = svm_classifier.predict(cnn_features_test)

# Performansı değerlendirme
accuracy = accuracy_score(np.argmax(test_labels, axis=1), svm_pred_labels)
print("SVM Model Accuracy:", accuracy)

# Karmaşıklık Matrisi
conf_matrix_svm = confusion_matrix(np.argmax(test_labels, axis=1), svm_pred_labels)

# Karmaşıklık matrisini seaborn kullanarak görselleştir
class_names = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_svm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Karmaşıklık Matrisi (SVM)')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.show()

# Sınıflandırma Raporu
print(classification_report(np.argmax(test_labels, axis=1), svm_pred_labels, target_names=class_names, zero_division=1))