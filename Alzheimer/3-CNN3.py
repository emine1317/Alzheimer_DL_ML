import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import BatchNormalization
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

# Veriyi eğitim, test ve doğrulama setlerine ayır
train_data, temp_data, train_labels, temp_labels = train_test_split(data, labels_one_hot, test_size=0.2, random_state=42)
test_data, validation_data, test_labels, validation_labels = train_test_split(temp_data, temp_labels, test_size=0.5, random_state=42)

# Piksel değerlerini 0 ile 1 arasında normalize et
train_data = train_data / 255.0
test_data = test_data / 255.0
validation_data = validation_data / 255.0

# CNN modelini oluştur 95312-- 98438
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

cnn_model.summary()
# Modeli derleme
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Veriyi yeniden şekillendirme
train_data_reshaped = train_data.reshape((train_data.shape[0], 64,64, 1))
test_data_reshaped = test_data.reshape((test_data.shape[0],64,64, 1))
validation_data_reshaped = validation_data.reshape((validation_data.shape[0],64,64, 1))

# Early stopping callback'i oluşturma
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Modeli eğitme
history = cnn_model.fit(train_data_reshaped, train_labels, epochs=10, batch_size=32, validation_data=(validation_data_reshaped, validation_labels), callbacks=[early_stopping])

# Plotting training and validation curves
plt.figure(figsize=(12, 5))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()

# Doğruluk grafiği
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Val_Accuracy', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

# Kayıp grafiği
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='loss', color='green')
plt.plot(history.history['val_loss'], label='Val_Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()


# Modelin performansını değerlendirme
cnn_scores = cnn_model.evaluate(test_data_reshaped, test_labels)
print(f"CNN Model Accuracy: {cnn_scores[1]}")

# Karmaşıklık Matrisi
labels_pred_one_hot = cnn_model.predict(test_data_reshaped)
labels_pred_flat = np.argmax(labels_pred_one_hot, axis=1)
conf_matrix_cnn = confusion_matrix(np.argmax(test_labels, axis=1), labels_pred_flat)

# Karmaşıklık matrisini seaborn kullanarak görselleştir
class_names = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_cnn, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix (CNN)')
plt.xlabel('Predicted Values')
plt.ylabel('Actuel Values')
plt.show()

# Sınıflandırma Raporu
class_report = classification_report(np.argmax(test_labels, axis=1), labels_pred_flat, target_names=class_names, zero_division=1, digits=5)
print(class_report)

from sklearn.metrics import f1_score, cohen_kappa_score

# Modelin performansını değerlendirme
cnn_scores = cnn_model.evaluate(test_data_reshaped, test_labels)
cnn_accuracy = cnn_scores[1]

# F1 skoru hesaplama
cnn_predictions = np.argmax(cnn_model.predict(test_data_reshaped), axis=1)
cnn_f1 = f1_score(np.argmax(test_labels, axis=1), cnn_predictions, average='weighted')

# Kappa katsayısı hesaplama
cnn_kappa = cohen_kappa_score(np.argmax(test_labels, axis=1), cnn_predictions)

# Modelin performansını yazdırma
print(f"CNN Model Accuracy: {cnn_accuracy}")
print(f"CNN Model F1 Score: {cnn_f1}")
print(f"CNN Model Kappa Score: {cnn_kappa}")