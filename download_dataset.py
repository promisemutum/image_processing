import os
import requests
from io import BytesIO
from datasets import load_dataset
from PIL import Image

def download_images(start_index=50, end_index=150, output_dir='ground_truth'):
    print(f"Streaming images {start_index} to {end_index} from kaupane/vintage-photography-captions...")
    
    dataset = load_dataset("kaupane/vintage-photography-captions", split="train", streaming=True)
    os.makedirs(output_dir, exist_ok=True)
    
    count = 0
    downloaded = 0
    total_to_download = end_index - start_index + 1
    
    for item in dataset:
        count += 1
        if count < start_index:
            continue
        if count > end_index:
            break
            
        if 'image_url' in item and item['image_url']:
            url = item['image_url']
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
                
                # Make sure to handle format issues
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                    
                file_name = f"vintage_{count:03d}.jpg"
                file_path = os.path.join(output_dir, file_name)
                image.save(file_path, "JPEG")
                
                downloaded += 1
                print(f"[{downloaded}/{total_to_download}] Saved {file_path} (Index: {count})")
            except Exception as e:
                print(f"Failed to download image (Index: {count}) from {url}: {e}")
        else:
            print(f"Skipping item because missing image_url (Index: {count})")

    print(f"\nSuccessfully downloaded {downloaded} images to '{output_dir}/'.")

if __name__ == "__main__":
    download_images(start_index=50, end_index=150)
