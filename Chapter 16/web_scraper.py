
import urllib.parse

def get_product_listings(query):
    """
    Redirect-only search: returns search URLs for Amazon, Flipkart, and eBay.
    """
    encoded_query = urllib.parse.quote_plus(query.strip())

    return [
        {
            "name": "Amazon Search",
            "price": "N/A",
            "merchant": "Amazon",
            "link": f"https://www.amazon.in/s?k={encoded_query}"
        },
        {
            "name": "Flipkart Search",
            "price": "N/A",
            "merchant": "Flipkart",
            "link": f"https://www.flipkart.com/search?q={encoded_query}"
        },
        {
            "name": "eBay Search",
            "price": "N/A",
            "merchant": "eBay",
            "link": f"https://www.ebay.com/sch/i.html?_nkw={encoded_query}"
        }
    ]
