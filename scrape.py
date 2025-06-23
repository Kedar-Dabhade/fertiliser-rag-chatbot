import requests
from bs4 import BeautifulSoup
import pandas as pd
from PIL import Image
import pytesseract
from io import BytesIO
import re
import os

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

BASE_URL = "https://resupply.co.nz"
PAGES = [
    f"{BASE_URL}/collections/fertiliser",
    f"{BASE_URL}/collections/fertiliser?page=2"
]

def extract_nutrient_levels(image_url):
    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        text = pytesseract.image_to_string(image)
        pattern = r"([A-Za-z]{1,3})[\s:]+([\d.]+)%"
        nutrients = dict(re.findall(pattern, text))
        return nutrients
    except Exception as e:
        print(f"OCR failed for {image_url}: {e}")
        return {}

def get_all_product_links():
    links = set()
    for page_url in PAGES:
        res = requests.get(page_url, headers=HEADERS)
        soup = BeautifulSoup(res.text, "html.parser")
        for tag in soup.select("a.product-item__title"):
            href = tag.get("href")
            if href and "/products/" in href:
                full_url = BASE_URL + href
                links.add(full_url)
    return list(links)

def extract_section_text(soup, label_text):
    try:
        headers = soup.find_all("div", class_="toggle__heading")
        for header in headers:
            span = header.find("span")
            if span and label_text.lower() in span.get_text(strip=True).lower():
                content = header.find_next_sibling("div", class_="toggle__content")
                if content:
                    return content.get_text(separator="\n", strip=True)
    except Exception as e:
        print(f"Error parsing {label_text} section: {e}")
    return ""


def scrape_product(url):
    res = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(res.text, "html.parser")

    name = soup.find("h1", class_="product__title")
    price = soup.find("strong", id="fertiliser-product-price--product-price")
    unit = soup.find("span", id="fertiliser-product-price--measurement")

    description = soup.select_one(".product__description")
    benefits = soup.select_one("#why-use-product-section")

    # Nutrient image
    nutrient_img_url = ""
    for img in soup.find_all("img"):
        src = img.get("src", "")
        alt = img.get("alt", "").lower()
        if "nutrient" in alt or "nutrient" in src.lower():
            nutrient_img_url = src
            if nutrient_img_url.startswith("//"):
                nutrient_img_url = "https:" + nutrient_img_url
            elif nutrient_img_url.startswith("/"):
                nutrient_img_url = BASE_URL + nutrient_img_url
            break

    nutrients = extract_nutrient_levels(nutrient_img_url) if nutrient_img_url else {}

    return {
        "Name": name.text.strip() if name else "",
        "Price": price.text.strip() if price else "",
        "Unit": unit.text.strip() if unit else "",
        "URL": url,
        "Description": description.get_text(separator="\n", strip=True) if description else "",
        "Benefits": benefits.get_text(separator="\n", strip=True) if benefits else "",
        "Application & Advice": extract_section_text(soup, "Applications and Advice"),
        "Storage": extract_section_text(soup, "Storage"),
        "Safety, Mixing and Compatibility": extract_section_text(soup, "Safety, Mixing and Compatibility"),
        "Nutrient Levels": nutrients
    }

def main():
    print("Fetching product links...")
    product_links = get_all_product_links()
    print(f"Found {len(product_links)} products.")

    data = []
    for link in product_links:
        print(f"Scraping: {link}")
        try:
            item = scrape_product(link)
            data.append(item)
        except Exception as e:
            print(f"❌ Failed: {link}\nError: {e}")

    if not data:
        print("❌ No data scraped.")
        return

    df = pd.json_normalize(data)

    # Expand nutrient levels
    if "Nutrient Levels" in df.columns:
        df["Nutrient Levels"] = df["Nutrient Levels"].apply(lambda x: x if isinstance(x, dict) else {})
        nutrients_df = df["Nutrient Levels"].apply(pd.Series)

        # Ensure consistent columns
        all_nutrients = ['N', 'P', 'K', 'S', 'Ca', 'Mg', 'B', 'Zn', 'Mo', 'Cu', 'Fe', 'Mn']
        for col in all_nutrients:
            if col not in nutrients_df.columns:
                nutrients_df[col] = None

        nutrients_df = nutrients_df[all_nutrients]
        nutrients_df.columns = [f"Nutrient_{col}" for col in nutrients_df.columns]

        df.drop(columns=["Nutrient Levels"], inplace=True)
        df = pd.concat([df, nutrients_df], axis=1)

    output_file = "resupply_fertilisers_full3.xlsx"
    df.to_excel(output_file, index=False)
    print(f"\n✅ Data saved to: {os.path.abspath(output_file)}")

if __name__ == "__main__":
    main()
