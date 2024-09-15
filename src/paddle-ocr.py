import multiprocessing
from paddleocr import PaddleOCR
import numpy as np
import cv2
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import requests

# Initialize PaddleOCR for CPU
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)

def fetch_image(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return np.frombuffer(response.content, np.uint8)
    except Exception as e:
        print(f"Error fetching image from {url}: {e}")
    return None

def process_image(image_data):
    if image_data is not None:
        try:
            img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            result = ocr.ocr(img)
            # Extract text from OCR result
            text = ' '.join([line[1][0] for line in result[0]])
            return text
        except Exception as e:
            print(f"Error processing image: {e}")
    return ''

def process_batch(urls):
    with ThreadPoolExecutor() as executor:
        image_data = list(executor.map(fetch_image, urls))
   
    with multiprocessing.Pool() as pool:
        results = pool.map(process_image, image_data)
   
    return results

def process_dataframe(df, batch_size=20):
    results = []
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i+batch_size]
       
        # Process batch
        batch_results = process_batch(batch_df['image_link'].tolist())
       
        results.extend(batch_results)
       
        print(f"Processed batch {i//batch_size + 1}/{len(df)//batch_size + 1}")
   
    return results

def run_ocr(input_file, output_file, start_row=None, end_row=None):
    # Read the input DataFrame
    df = pd.read_csv(input_file)
   
    # Initialize the 'OCR_text' column if not present
    if 'OCR_text' not in df.columns:
        df['OCR_text'] = ''
   
    # If specific rows are provided, slice the DataFrame for processing
    if start_row is not None and end_row is not None:
        df_slice = df.iloc[start_row:end_row]
        # Process only the sliced part of the DataFrame
        ocr_results = process_dataframe(df_slice)
        # Update the OCR results in the full DataFrame
        df.loc[start_row:end_row-1, 'OCR_text'] = ocr_results
    else:
        # If no range is specified, process the whole DataFrame
        ocr_results = process_dataframe(df)
        df['OCR_text'] = ocr_results
   
    # Save the updated full DataFrame with results
    df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")
   
    # Return the full DataFrame with OCR results
    return df

if __name__ == '__main__':
    input_file = "../dataset/test.csv"
    output_file = "output_data_with_ocr.csv"
   
    # Define the rows you want to process (e.g., from row 100 to row 200)
    start_row = 60000
    end_row = 70000
   
    # Run OCR on the specified rows and return the full DataFrame
    processed_df = run_ocr(input_file, output_file, start_row=start_row, end_row=end_row)
    print(processed_df)