# Importing libraries
import random
import os
import numpy as np
import pandas as pd 
import seaborn as sns
from PIL import Image, ImageFilter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Function to read and flatten images
def read_process_and_flatten_image(file_path, target_size=(350, 350), blur_radius=2):
    with Image.open(file_path) as img:
        # Resize image
        img = img.resize(target_size, Image.ANTIALIAS)
        # Apply blurring
        img = img.filter(ImageFilter.GaussianBlur(blur_radius))
        # Normalize pixel values
        img_array = np.array(img) / 255.0
        # Flatten image
        img_array_flattened = img_array.flatten()
    return img_array_flattened

# Define the directories containing the images
train_dir = "train"

# Initialize lists to store file paths, pixel values, and labels
file_paths = []
pixels = []
labels = []

# Iterate over the train directory to collect file paths, pixel values, and labels
for file_name in os.listdir(train_dir)[:1100]:  # Selecting 1000 images for training
    file_path = os.path.join(train_dir, file_name)
    file_paths.append(file_path)
    image_pixels = read_process_and_flatten_image(file_path)
    pixels.append(image_pixels)
    labels.append(file_name.split('.')[0])  # Assuming label is before the first '.'

# Apply PCA to reduce dimensionality of the pixel values
pca = PCA(n_components=100)  # Adjust the number of components as needed
pixels_pca = pca.fit_transform(pixels)

# Create a DataFrame from the collected data
data = pd.DataFrame({'file_path': file_paths, 'pixels_pca': pixels_pca.tolist(), 'label': labels})

# Shuffle the DataFrame to randomize the order of the data
data = data.sample(frac=1).reset_index(drop=True)


data['label'].value_counts()

sns.set(style="whitegrid")  # Set the style
plt.figure(figsize=(10, 6))  # Set the figure size

custom_cmap = "viridis"  # You can choose any colormap you prefer

ax = sns.countplot(x='label', data=data, order=data['label'].value_counts().index, palette=custom_cmap)

for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                textcoords='offset points')

plt.xlabel('Label', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.title('Count of Label', fontsize=16)
plt.xticks(rotation=45)
plt.show()

# Dog Image
plt.figure(figsize=(15, 6))
plt.suptitle('Random Dog Images', fontsize=16)
for i, img_file in enumerate(random.sample(list(data[data['label'] == 'dog']['file_path']), 5), 1):
    plt.subplot(2, 5, i)
    img = mpimg.imread(img_file)
    plt.imshow(img)
    plt.axis('off')


## Cat Image    
plt.figure(figsize=(15, 6))
plt.suptitle('Random Cat Images', fontsize=16)
for i, img_file in enumerate(random.sample(list(data[data['label'] == 'cat']['file_path']), 5), 1):
    img_path = os.path.join("/kaggle/working/train", img_file)
    plt.subplot(2, 5, i)
    img = mpimg.imread(img_path)
    plt.imshow(img)
    plt.axis('off')

plt.show()

# Splitting data for the Training model

X_train, X_test, y_train, y_test = train_test_split(data['pixels_pca'].tolist(), data['label'], test_size=0.1, random_state=42)

print('Training data ', len(X_train))
print('Testing data ', len(X_test))


# Standardize the pixel values
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pd.DataFrame(X_train_scaled)

# Fitting the model
svm_clf = SVC()
svm_clf.fit(X_train_scaled, y_train)
svm_pred = svm_clf.predict(X_test_scaled)

# Evaluation of the model
svm_accuracy = accuracy_score(y_test, svm_pred)
print("SVM Accuracy:", svm_accuracy)
print("SVM Classification Report:")
# Classification Report
print(classification_report(y_test, svm_pred))

