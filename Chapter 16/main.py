
import os
import logging
from image_utils import extract_text_from_image
from search_utils import generate_search_query
from web_scraper import get_product_listings
from summarizer import summarize_product_pages

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def find_first_image_in_assets():
    assets_folder = "assets"
    if not os.path.exists(assets_folder):
        raise FileNotFoundError(f"Assets folder '{assets_folder}' not found")

    for file in os.listdir(assets_folder):
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            return os.path.join(assets_folder, file)
    raise FileNotFoundError("No image found in the assets folder.")

def main():
    try:
        image_path = find_first_image_in_assets()
        logging.info(f"Using image: {image_path}")

        extracted_text = extract_text_from_image(image_path)
        if not extracted_text:
            raise ValueError("No text could be extracted from the image")
        logging.info(f"OCR Extracted Text:\n{extracted_text}")

        query = generate_search_query(extracted_text)
        if not query:
            raise ValueError("Failed to generate a valid search query")

        query = query.replace('"', '').replace("₹", "rs").replace("or less", "under").replace("alternative", "")
        logging.info(f"Search Query:\n{query}")

        results = get_product_listings(query)
        if not results:
            logging.warning("No product listings found")
            return

        print("\nHere are some similar or better alternatives you can check out:\n")
        for i, res in enumerate(results, 1):
            print(f"{i}. {res['name']}")
            print(f"   Price: {res['price']}")
            print(f"   Store: {res['merchant']}")
            print(f"   Link: {res['link']}\n")

        # Summarize the product pages
        print("\nSummary of Product Listings:\n")
        summaries = summarize_product_pages(results)
        for s in summaries:
            print(f"{s['store']} Summary:\n{s['summary']}\n")

    except FileNotFoundError as e:
        logging.error(f"File error: {str(e)}")
    except ValueError as e:
        logging.error(f"Value error: {str(e)}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()
