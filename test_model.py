import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load your trained model
model = load_model('retinopathy_model.h5')

# Define the path to your images
folder_path = 'C:/Users/LENOVO/Desktop/new_retinopathy/colored_images'  # <-- your folder

# Define your class names based on your training labels
class_names = ['Class0', 'Class1', 'Class2', 'Class3', 'Class4']  
# 🔥 Replace these with your actual class names like ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

# Prepare a list to store results
results = []

# Loop through images
for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        img_path = os.path.join(folder_path, filename)
        
        # Load and preprocess the image
        img = Image.open(img_path).resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        prediction = model.predict(img_array)
        
        # Get the index of highest probability
        predicted_class_index = np.argmax(prediction)
        predicted_class_name = class_names[predicted_class_index]
        
        # Save the filename and prediction
        results.append({
            'filename': filename,
            'predicted_label': predicted_class_name,
            'confidence': float(np.max(prediction))  # optional: model confidence
        })

# Create a dataframe
df = pd.DataFrame(results)

# Save to CSV
df.to_csv('predictions.csv', index=False)

print("✅ Predictions saved to predictions.csv")
