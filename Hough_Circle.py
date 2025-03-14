import cv2
import pandas as pd
import numpy as np

# Constants
csv_path = "D:/MAJOR Projects/crater_coordinates.csv"
image_path = "D:/Sample Dataset/Luna-1/crater_images/654.png"  

def detect_craters(image_path, threshold=0.7):
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Apply Histogram Equalization to improve contrast
    img_equalized = cv2.equalizeHist(img)

    # Apply Adaptive Thresholding to convert to binary image
    img_thresholded = cv2.adaptiveThreshold(
        img_equalized, 
        255, 
        cv2.ADAPTIVE_THRESH_MEAN_C, 
        cv2.THRESH_BINARY, 
        11,  # Block size
        2   # Constant subtracted from the mean
    )

    # Apply Hough Circle Transform to detect craters
    detected_circles = cv2.HoughCircles(
        img_thresholded,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=30,  # Distance between circle centers
        param1=50,  # Higher sensitivity
        param2=threshold * 80,  # Threshold for detecting circles
        minRadius=10,  # Minimum radius of the circle
        maxRadius=80  # Maximum radius of the circle
    )
    
    # Check if any circles were detected
    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))  # Round the values
        return detected_circles[0]  # Return the detected circles
    print("No craters detected.")
    return []

def process_crater_image(image_path):
    # Read the existing crater data from the CSV file
    try:
        crater_data = pd.read_csv(csv_path)
    except FileNotFoundError:
        crater_data = pd.DataFrame(columns=["Image_Path", "Latitude", "Longitude", "Threshold", "Token"])

    # Check if the image path already exists in the database
    existing_entry = crater_data[crater_data["Image_Path"] == image_path]
    if not existing_entry.empty:
        # Retrieve and print the existing crater details
        latitude = existing_entry.iloc[0]["Latitude"]
        longitude = existing_entry.iloc[0]["Longitude"]
        threshold = existing_entry.iloc[0]["Threshold"]
        print(f"Crater already searched: Latitude={latitude}, Longitude={longitude}, Threshold={threshold}")
        return
    
    # Detect craters if the image path is not found
    detected_craters = detect_craters(image_path)
    if len(detected_craters) == 0:
        print("No craters detected.")
        return

    # Ensure the 'Token' column is numeric, convert if needed
    crater_data["Token"] = pd.to_numeric(crater_data["Token"], errors='coerce')  # 'coerce' turns non-numeric to NaN

    # Assign a new token
    max_token = crater_data["Token"].max() if not crater_data.empty else 0
    token = max_token + 1

    # Process detected craters
    for circle in detected_craters:
        x, y, radius = circle
        latitude = np.random.uniform(-90, 90)  # Replace with actual calculation
        longitude = np.random.uniform(-180, 180)  # Replace with actual calculation
        threshold_score = 0.7  # Example score
        
        # Create new row as a DataFrame
        new_row = pd.DataFrame([{
            "Image_Path": image_path,
            "Latitude": latitude,
            "Longitude": longitude,
            "Threshold": threshold_score,
            "Token": token
        }])
        
        # Append new row using pd.concat
        crater_data = pd.concat([crater_data, new_row], ignore_index=True)
        
        # Print detected crater details
        print(f"Detected Crater: Latitude={latitude}, Longitude={longitude}, Radius={radius}px, Threshold={threshold_score}")

    # Save the updated crater data to the CSV file
    crater_data.to_csv(csv_path, index=False)
    print(f"New crater detection saved to {csv_path}")

# Run the function
process_crater_image(image_path)
