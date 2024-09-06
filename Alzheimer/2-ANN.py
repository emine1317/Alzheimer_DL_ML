import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix,roc_curve, auc,classification_report, confusion_matrix, accuracy_score

def apply_sobel_filter(img):
    # Sobel filtresi uygula
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    filtered_img = np.sqrt(sobelx**2 + sobely**2)
    return filtered_img


def load_and_preprocess_data(folder_path, label):
    data = []
    labels = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (244, 244))
        # Bilateral filtresi uygulama
        img = cv2.bilateralFilter(img, 9, 75, 75)  # Daha fazla gürültü azaltma için parametreleri ayarlayabilirsiniz
        img = apply_sobel_filter(img) 
        data.append(img)
        labels.append(label)
    return data, labels
 # Resmi ekrana yazdırma
       
# Verileri yükleme ve etiketleri oluşturma
mild_demented_data, mild_demented_labels = load_and_preprocess_data('C:/Users/Casper/Desktop/Makine_Ogrenmesi/Alzheimer/Dataset/Mild_Demented', 0)
moderate_demented_data, moderate_demented_labels = load_and_preprocess_data('C:/Users/Casper/Desktop/Makine_Ogrenmesi/Alzheimer/Dataset/Moderate_Demented', 1)
non_demented_data, non_demented_labels = load_and_preprocess_data('C:/Users/Casper/Desktop/Makine_Ogrenmesi/Alzheimer/Dataset/Non_Demented', 2)
very_mild_demented_data, very_mild_demented_labels = load_and_preprocess_data('C:/Users/Casper/Desktop/Makine_Ogrenmesi/Alzheimer/Dataset/Very_Mild_Demented', 3)


# Verileri dört sınıfa göre ayır
mild_train_data, mild_test_data, mild_train_labels, mild_test_labels = train_test_split(mild_demented_data, mild_demented_labels, test_size=0.2, random_state=42)
moderate_train_data, moderate_test_data, moderate_train_labels, moderate_test_labels = train_test_split(moderate_demented_data, moderate_demented_labels, test_size=0.2, random_state=42)
non_train_data, non_test_data, non_train_labels, non_test_labels = train_test_split(non_demented_data, non_demented_labels, test_size=0.2, random_state=42)
very_mild_train_data, very_mild_test_data, very_mild_train_labels, very_mild_test_labels = train_test_split(very_mild_demented_data, very_mild_demented_labels, test_size=0.2, random_state=42)

# Validation verilerini ayrı olarak al
mild_train_data, mild_validation_data, mild_train_labels, mild_validation_labels = train_test_split(mild_train_data, mild_train_labels, test_size=0.25, random_state=42)
moderate_train_data, moderate_validation_data, moderate_train_labels, moderate_validation_labels = train_test_split(moderate_train_data, moderate_train_labels, test_size=0.25, random_state=42)
non_train_data, non_validation_data, non_train_labels, non_validation_labels = train_test_split(non_train_data, non_train_labels, test_size=0.25, random_state=42)
very_mild_train_data, very_mild_validation_data, very_mild_train_labels, very_mild_validation_labels = train_test_split(very_mild_train_data, very_mild_train_labels, test_size=0.25, random_state=42)

# Train verilerini birleştir
train_data = np.concatenate((mild_train_data, moderate_train_data, non_train_data, very_mild_train_data), axis=0)
train_labels = np.concatenate((mild_train_labels, moderate_train_labels, non_train_labels, very_mild_train_labels), axis=0)

# Validation verilerini birleştir
validation_data = np.concatenate((mild_validation_data, moderate_validation_data, non_validation_data, very_mild_validation_data), axis=0)
validation_labels = np.concatenate((mild_validation_labels, moderate_validation_labels, non_validation_labels, very_mild_validation_labels), axis=0)

# Test verilerini birleştir
test_data = np.concatenate((mild_test_data, moderate_test_data, non_test_data, very_mild_test_data), axis=0)
test_labels = np.concatenate((mild_test_labels, moderate_test_labels, non_test_labels, very_mild_test_labels), axis=0)

# Etiketleri one-hot encoding'e çevir
train_labels_one_hot = to_categorical(train_labels, num_classes=4)
validation_labels_one_hot = to_categorical(validation_labels, num_classes=4)
test_labels_one_hot = to_categorical(test_labels, num_classes=4)

# Verileri normalize et
train_data = train_data / 255.0
validation_data = validation_data / 255.0
test_data = test_data / 255.0

# Model oluştur
model_ann = Sequential([
    Flatten(input_shape=(244, 244)),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.25),
    Dense(4, activation='softmax')
])

model_ann.summary()

# Modeli derle
model_ann.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Modeli eğit
epochs = 5
history_ann = model_ann.fit(train_data, train_labels_one_hot, epochs=epochs, validation_data=(validation_data, validation_labels_one_hot))

# Eğitim sonuçlarını görselleştir
plt.plot(history_ann.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history_ann.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.xlabel('Epoklar')
plt.ylabel('Doğruluk')
plt.legend()   
plt.show()

# Kayıp grafiği
plt.plot(history_ann.history['loss'], label='Eğitim Kaybı')
plt.plot(history_ann.history['val_loss'], label='Doğrulama Kaybı')
plt.xlabel('Epoklar')
plt.ylabel('Kayıp')
plt.legend()
plt.show()


# Doğruluk grafiği
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history_ann.history['accuracy'], label='Accuracy', color='blue')
plt.plot(history_ann.history['val_accuracy'], label='Val_Accuracy', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

# Kayıp grafiği
plt.subplot(1, 2, 2)
plt.plot(history_ann.history['loss'], label='Loss', color='green')
plt.plot(history_ann.history['val_loss'], label='Val_Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()

# Test verileri üzerinde tahmin yap
predictions = model_ann.predict(test_data)
predicted_labels = np.argmax(predictions, axis=1)

# Gerçek ve tahmini etiketler arasındaki confusion matrix'i oluştur
conf_matrix = confusion_matrix(test_labels, predicted_labels)

# Classification report'u al
classif_report = classification_report(test_labels, predicted_labels, target_names=['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented'], digits=6)

# Confusion matrix'i görselleştir
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented'], 
            yticklabels=['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented'])
plt.xlabel('Predicted Value')
plt.ylabel('Actuel Value')
plt.title('Confusion Matrix (ANN)')
plt.show()

# Classification report'u yazdır
print("\nClassification Report:\n", classif_report)

# Doğruluk
accuracy = accuracy_score(test_labels, predicted_labels)
print("Doğruluk:", accuracy)

# F1 skoru
f1 = f1_score(test_labels, predicted_labels, average='weighted')
print("F1 Skoru:", f1)

# Kappa katsayısı
kappa = cohen_kappa_score(test_labels, predicted_labels)
print("Kappa Katsayısı:", kappa)
