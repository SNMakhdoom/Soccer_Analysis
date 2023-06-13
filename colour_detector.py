import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

colors = {
    'red': [255, 0, 0],
    'green': [0, 255, 0],
    'blue': [0, 0, 255],
    'yellow': [255, 255, 0]
}

data = []
target = []

for color, rgb in colors.items():
    for _ in range(1000): 
        r = rgb[0] + np.random.randint(-30, 30) 
        g = rgb[1] + np.random.randint(-30, 30) 
        b = rgb[2] + np.random.randint(-30, 30) 
        data.append([r, g, b])
        target.append(color)


data = np.array(data)
target = np.array(target)

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Test the model
print('Model accuracy:', knn.score(X_test, y_test))

# Now you can use this model to predict the color of a pixel in an image
image = cv2.imread('test.jpg')
pixel_rgb = image[20, 30]  # get the RGB value of the pixel at position (20, 30)
predicted_color = knn.predict([pixel_rgb])
print('Predicted color:', predicted_color)
