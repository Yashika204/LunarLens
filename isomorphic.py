import cv2
import numpy as np
import pandas as pd
import uuid  # For generating unique tokens

# Function to detect craters
def detect_craters(img_path, csv_path):
    # Load the image
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Edge detection using Canny
    edges = cv2.Canny(gray, 100, 200)
    
    # Morphological operations to close gaps in edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Find contours of the closed edges
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Prepare a DataFrame to store crater information
    crater_data = []

    for contour in contours:
        # Approximate the contour to a circle
        center, radius = cv2.minEnclosingCircle(contour)
        center = tuple(map(int, center))
        radius = int(radius)
        
        # Filter out small detections
        if radius > 10:  # Adjust this threshold based on your dataset
            # Draw the detected crater on the image
            cv2.circle(img, center, radius, (0, 255, 0), 2)
            
            # Generate a unique token for the crater
            token = str(uuid.uuid4())
            
            # Append crater details to the DataFrame
            crater_data.append({
                'Token': token,
                'Center_X': center[0],
                'Center_Y': center[1],
                'Radius': radius
            })
    
    # Save the crater data to a CSV file
    crater_df = pd.DataFrame(crater_data)
    
    if csv_path:  # Append to the existing CSV or create a new one
        try:
            existing_data = pd.read_csv(csv_path)
            crater_df = pd.concat([existing_data, crater_df], ignore_index=True)
        except FileNotFoundError:
            pass  # If CSV doesn't exist, a new one will be created
    
    crater_df.to_csv(csv_path, index=False)
    print(f"Crater data saved to {csv_path}")
    
    # Show the final image with detected craters
    cv2.imshow("Detected Craters", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
img_path ="D:/Sample Dataset/Luna-1/crater_images/78.png"  # Replace with your image path
csv_path = "D:/MAJOR Projects/crater_coordinates.csv"         # Replace with your desired CSV path
detect_craters(img_path, csv_path)
