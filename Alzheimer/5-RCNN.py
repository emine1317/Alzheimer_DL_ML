import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
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

# Veriyi eğitim, test ve doğrulama setlerine ayır
train_data, test_val_data, train_labels, test_val_labels = train_test_split(data, labels_one_hot, test_size=0.3, random_state=42)
test_data, val_data, test_labels, val_labels = train_test_split(test_val_data, test_val_labels, test_size=0.5, random_state=42)

# Piksel değerlerini 0 ile 1 arasında normalize et
train_data = train_data / 255.0
test_data = test_data / 255.0
val_data = val_data / 255.0

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

def rcnn_model(num_classes):
    try:
        # Define the input layer
        input_layer = Input(shape=(64, 64, 1))  # Grayscale image
        
       # Define the CNN layers for feature extraction
        cnn_layer1 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1')(input_layer)
        cnn_layer2 = MaxPooling2D((2, 2), name='maxpool1')(cnn_layer1)
        cnn_layer3 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2')(cnn_layer2)
        cnn_layer4 = MaxPooling2D((2, 2), name='maxpool2')(cnn_layer3)
        cnn_layer5 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3')(cnn_layer4)
        cnn_layer6 = MaxPooling2D((2, 2), name='maxpool3')(cnn_layer5)
        cnn_layer7 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4')(cnn_layer6)
        cnn_layer8 = MaxPooling2D((2, 2), name='maxpool4')(cnn_layer7)
        cnn_layer9 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5')(cnn_layer8)
        cnn_layer10 = MaxPooling2D((2, 2), name='maxpool5')(cnn_layer9)
        cnn_output = Flatten(name='flatten')(cnn_layer10)

        # Define the RPN layers for object proposals
        rpn_layer1 = Dense(64, activation='relu', name='rpn_dense1')(cnn_output)
        rpn_layer2 = Dense(128, activation='relu', name='rpn_dense2')(rpn_layer1)
        rpn_output = Dense(4, activation='sigmoid', name='rpn_output')(rpn_layer2)

        # Define the classification layers
        classification_layer1 = Dense(512, activation='relu', name='cls_dense1')(cnn_output)
        classification_layer2 = Dense(512, activation='relu', name='cls_dense2')(classification_layer1)
        classification_output = Dense(num_classes, activation='softmax', name='cls_output')(classification_layer2)

        # Define the RCNN model
        rcnn_model = tf.keras.Model(inputs=input_layer, outputs=[rpn_output, classification_output])
        
        # Compile the model
        rcnn_model.compile(optimizer='adam', loss=['mse', 'categorical_crossentropy'], metrics=['accuracy'])
        
        return rcnn_model
    except Exception as e:
        # Log the error
        print(f"Error: {e}")
        return None

# Instantiate RCNN model
rcnn = rcnn_model(num_classes=4)

# Print summary
rcnn.summary()

# Veriyi yeniden şekillendirme
train_data_reshaped = train_data.reshape((train_data.shape[0], 64, 64, 1))
test_data_reshaped = test_data.reshape((test_data.shape[0], 64, 64, 1))
val_data_reshaped = val_data.reshape((val_data.shape[0], 64, 64, 1))

# Early stopping callback'i oluşturma
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Modeli eğitme
#history = rcnn.fit(train_data_reshaped, [train_labels, train_labels], epochs=15, batch_size=32, validation_data=(val_data_reshaped, [val_labels, val_labels]))

# Modeli eğitme
history = rcnn.fit(train_data_reshaped, [train_labels, train_labels], epochs=15, batch_size=32, validation_data=(val_data_reshaped, [val_labels, val_labels]))
print(history.history.keys())

# Tahminleri yapma
predictions = rcnn.predict(test_data_reshaped)
y_pred_classes = np.argmax(predictions[1], axis=1)
y_true = np.argmax(test_labels, axis=1)

# Doğruluk grafiği
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['rpn_output_accuracy'], label='rpn_output_accuracy', color='blue')
plt.plot(history.history['val_rpn_output_accuracy'], label='val_rpn_output_accuracy', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

# Kayıp grafiği
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Loss', color='green')
plt.plot(history.history['val_loss'], label='Val_Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()


# Confusion matrix hesaplama
confusion_mat = confusion_matrix(y_true, y_pred_classes)

# Confusion matrix'i görselleştirme
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.xlabel('Predicted Values')
plt.ylabel('Actuel Values')

plt.title('Confusion Matrix (RCNN)')
plt.show()

# Accuracy'yi yazdırma
accuracy = accuracy_score(y_true, y_pred_classes)
print("Accuracy:", accuracy)

# Sınıflandırma raporu
class_names = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']
print(classification_report(y_true, y_pred_classes, target_names=class_names, zero_division=1,digits=6))

# Doğruluk
accuracy = accuracy_score(y_true, y_pred_classes)
print("Doğruluk:", accuracy)

# F1 skoru
f1 = f1_score(y_true, y_pred_classes, average='weighted')
print("F1 Skoru:", f1)

# Kappa katsayısı
kappa = cohen_kappa_score(y_true, y_pred_classes)
print("Kappa Katsayısı:", kappa)
