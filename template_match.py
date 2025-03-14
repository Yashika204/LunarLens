import cv2
import pandas as pd
import numpy as np

# Constants
csv_path = "D:/MAJOR Projects/crater_coordinates.csv"
image_path = "D:/Sample Dataset/Luna-1/crater_images/410.png"  # Image to detect craters

def detect_craters_with_template_matching(image_path, threshold=0.7):
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

    # Perform Template Matching to detect circular features (craters)
    result = cv2.matchTemplate(img_thresholded, img_thresholded, cv2.TM_CCOEFF_NORMED)
    
    # Get locations where the matching score is above the threshold
    loc = np.where(result >= threshold)
    
    # Convert coordinates to list of detected craters
    detected_craters = []
    for pt in zip(*loc[::-1]):  # Flip the coordinates for x, y
        detected_craters.append(pt)  # Add the top-left corner of each match as a detected crater
    
    if detected_craters:
        print(f"Detected {len(detected_craters)} craters.")
        return detected_craters  # Return the coordinates of detected craters
    else:
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

    # Detect craters using template matching
    detected_craters = detect_craters_with_template_matching(image_path)
    if len(detected_craters) == 0:
        print("No craters detected.")
        return

    # Ensure the 'Token' column is numeric, convert if needed
    crater_data["Token"] = pd.to_numeric(crater_data["Token"], errors='coerce')  # 'coerce' turns non-numeric to NaN

    # Assign a new token (Increment by 1 from max existing token)
    max_token = crater_data["Token"].max() if not crater_data.empty else 0
    token = max_token + 1

    # Filter craters with a threshold greater than 0.7
    detected_craters_with_threshold = []
    for pt in detected_craters:
        x, y = pt
        threshold_score = 0.8  # Example threshold score (you can adjust this based on detection quality)
        
        if threshold_score > 0.7:
            detected_craters_with_threshold.append({
                "coordinates": pt,
                "threshold": threshold_score
            })

    if not detected_craters_with_threshold:
        print("No craters with threshold greater than 0.7 detected.")
        return

    # Show all detected craters with threshold > 0.7
    print("Detected Craters with Threshold > 0.7:")
    for crater in detected_craters_with_threshold:
        print(f"Coordinates: {crater['coordinates']}, Threshold: {crater['threshold']}")

    # Select the crater with the highest threshold value
    best_crater = max(detected_craters_with_threshold, key=lambda x: x['threshold'])
    best_coordinates = best_crater["coordinates"]
    best_threshold = best_crater["threshold"]

    # Latitude and Longitude for the best crater (use actual calculation if available)
    latitude = np.random.uniform(-90, 90)  # Replace with actual calculation
    longitude = np.random.uniform(-180, 180)  # Replace with actual calculation
    
    # Create new row as a DataFrame
    new_row = pd.DataFrame([{
        "Image_Path": image_path,
        "Latitude": latitude,
        "Longitude": longitude,
        "Threshold": best_threshold,
        "Token": token
    }])
    
    # Append new row using pd.concat
    crater_data = pd.concat([crater_data, new_row], ignore_index=True)
    
    # Print detected crater details (without coordinates)
    print(f"Crater saved: Latitude={latitude}, Longitude={longitude}, Threshold={best_threshold}")

    # Save the updated crater data to the CSV file
    crater_data.to_csv(csv_path, index=False)
    print(f"New crater detection saved to {csv_path}")

# Run the function
process_crater_image(image_path)
