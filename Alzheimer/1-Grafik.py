import tensorflow as tf 
import matplotlib.pyplot as plt 
import numpy as np 
import os 
import pathlib 
import random
import matplotlib.image as mpimg
from tensorflow.keras.utils import image_dataset_from_directory

#veri alma
path = 'C:/Users/iyagm/Anaconda/Tez/Dataset'
veri_küt = pathlib.Path(path)

#veri etiket isimleri
sınıf_isim = np.array([sorted(item.name for item in veri_küt.glob("*"))])
print(sınıf_isim)

#pasta grafiği
plt.rcParams["figure.figsize"] = (10,6)
plt.rcParams['figure.dpi'] = 300
colors = ["#0033CC", "#D85F9C", "#FFFF33", "#CC3333"]
class_dist = {}
def image_counter(folder_path):
    basename = os.path.basename(folder_path)
    print('\033[92m'+f"A search has been initiated within the folder named '{basename}'."+'\033[0m')
    image_extensions = ['.jpg', '.jpeg', '.png']

    for root, dirs, _ in os.walk(folder_path):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            count = 0

            for filename in os.listdir(dir_path):
                file_ext = os.path.splitext(filename)[1].lower()

                if file_ext in image_extensions:
                    count += 1
            
            class_dist[dir_name] = count
            print(f"There are \033[35m{count}\033[0m images in the {dir_name} folder.")
    print('\033[92m'+"The search has been completed."+'\033[0m')
    
    keys = list(class_dist.keys())
    values = list(class_dist.values())
    explode = (0.1,)*len(keys)
    
    labels = [f'{key} ({value} images)' for key, value in zip(keys, values)]
    
    plt.pie(values, explode=explode,labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90, colors=colors, textprops={'fontsize': 12, "fontweight" : "bold", "color":"black"},  wedgeprops=
           {'edgecolor':'darkblue'} , labeldistance=1.15)
    plt.title("Alzheimer MR Görüntülerinin Dağılımı", size=12, fontweight="bold")

image_counter(path)

#MR renkli görsel gösterme ve isimi
def sample_bringer(path, target, num_samples=5):
    # Belirtilen hedef (class) için tam dosya yolu oluşturulur.
    class_path = os.path.join(path, target)

    # Hedef dizinindeki tüm .jpg uzantılı görüntü dosyaları alınır.
    image_files = [image for image in os.listdir(class_path) if image.endswith('.jpg')]

    # Belirtilen sayıda örnek görüntüyü görselleştirmek için bir alt grafik penceresi oluşturulur.
    fig, ax = plt.subplots(1, num_samples, facecolor="gray")
    fig.suptitle(f'{target} Beyin MR Örnekleri', color="yellow", fontsize=16, fontweight='bold', y=0.75)
    
    # Belirtilen sayıda örnek görüntü döngüsü.
    for i in range(num_samples):
        # Görüntü dosyasının tam yolu oluşturulur.
        image_path = os.path.join(class_path, image_files[i])
        # Görüntü dosyası okunur.
        img = mpimg.imread(image_path)

        # Alt grafik penceresine görüntü eklenir.
        ax[i].imshow(img)
        ax[i].axis('off')
        ax[i].set_title(f'Örnek {i+1}', color="aqua")

    # Görüntüler arasındaki boşluğu düzenler.
    plt.tight_layout()

# class_names listesindeki her bir sınıf için örnek görüntüleri görselleştiren bir döngü.
for target in sınıf_isim[0]:
    sample_bringer(path, target=target)
'''
batch_size = 32
img_height = 180
img_width = 180
seed = 42

train_data = image_dataset_from_directory(
                  veri_küt,
                  validation_split=0.2,
                  subset="training",
                  seed=seed,
                  image_size=(img_height, img_width),
                  batch_size=batch_size)


val_data = image_dataset_from_directory(
                 veri_küt,
                 validation_split=0.2,
                 subset="validation",
                 seed=seed,
                 image_size=(img_height,img_width),
                 batch_size=batch_size)

from tensorflow.keras import layers

model = tf.keras.Sequential([
    
   layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
    
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
    
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
    
  layers.Dropout(0.5),
  layers.Flatten(),
    
  layers.Dense(128, activation='relu'),
  layers.Dense(4,activation="softmax")
])
model.compile(optimizer="Adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"])
epochs = 1
history = model.fit(train_data,
                    epochs=epochs,
                    validation_data=val_data, 
                    batch_size=batch_size)
'''

















