from PIL import Image
import cv2
import numpy as np
import glob
import pandas as pd
def find_exterior_edges(image_path):
    # Load the image
    img = Image.open(image_path)

    # Convert the image to grayscale
    gray_img = img.convert("L")

    # Convert PIL Image to numpy array
    np_img = np.array(gray_img)

    np_blurred = cv2.GaussianBlur(np_img, (3,3), 0)
    # apply morphology
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(np_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Apply erosion and d    
    # Create a mask for exterior edges
    exterior_edges_mask = np.zeros_like(binary)
   
    # Draw filled contours on the mask
    areas=[]
    for contour in contours:
        area = cv2.contourArea(contour)
        areas.append(area)
    
    q_10,q_5,q_1=pd.DataFrame(areas).quantile([0.1,0.05,0.01]).values
    for contour in contours:
        area = cv2.contourArea(contour)
        if (area >=q_5) :  # Adjust this threshold based on the size of the UML diagram components
            cv2.drawContours(exterior_edges_mask, [contour], -1, (255), thickness=cv2.FILLED)
    

    # Apply erosion and dilation for refining the contour
    kernel = np.ones((1, 1), np.uint8)
    exterior_edges_mask = cv2.erode(exterior_edges_mask, kernel, iterations=2)
    exterior_edges_mask = cv2.dilate(exterior_edges_mask, kernel, iterations=1)

    #  Smoothen the mask
    smooth_mask = cv2.medianBlur(exterior_edges_mask, 3)

    smooth_mask = cv2.GaussianBlur(smooth_mask, (3, 3), 0)

    return smooth_mask



def remove_background(image_path, edges_mask):

    # Load the image
    img = Image.open(image_path)

    # Convert the image to RGBA mode
    img = img.convert("RGBA")

    # Invert the edges mask
    inverted_edges_mask = 255 - edges_mask

    # Convert the inverted edges mask to PIL Image
    inverted_edges_img = Image.fromarray(inverted_edges_mask, mode='L')

    # Create a new image with alpha channel
    new_img = Image.new("RGBA", img.size, (255, 255, 255, 0))

    # Composite the original image and the inverted contour mask
    new_img.paste(img, (0, 0), img)
    new_img.paste((255, 255, 255, 0), (0, 0), inverted_edges_img)

    return new_img

image_dir='Images-2'
output_dir='Output-2'

for image_path in glob.glob(image_dir+"/**"):
    output_img=remove_background(image_path=image_path,edges_mask=find_exterior_edges(image_path))
    output_img.save(output_dir+f'/'+image_path.split('/')[-1].split('.')[0] +'.png')
    