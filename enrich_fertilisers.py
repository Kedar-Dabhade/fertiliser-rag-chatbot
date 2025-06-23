import pandas as pd
import openai
import time
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

INPUT_XLSX = "products_enriched_split.xlsx"
OUTPUT_XLSX = "products_enriched_final2.xlsx"
BATCH_SIZE = 3

COLUMNS = [
    "Name", "Price", "Unit", "URL", "Description", "Benefits", "Application & Advice", "Storage", "Safety, Mixing and Compatibility", "Nutrient_N", "Nutrient_P", "Nutrient_K", "Nutrient_S", "Nutrient_Ca", "Nutrient_Mg"
]

ENRICH_COLUMNS = [
    "Description", "Benefits", "Application & Advice", "Storage", "Safety, Mixing and Compatibility"
]
NUTRIENT_COLUMNS = [
    "Nutrient_N", "Nutrient_P", "Nutrient_K", "Nutrient_S", "Nutrient_Ca", "Nutrient_Mg"
]

PROMPT_TEMPLATE = '''
You are an agricultural domain expert and data enrichment assistant. Your task is to expand and enhance specific columns of fertilizer product data using trusted agricultural sources—especially from https://www.resupply.co.nz and https://www.ravensdown.co.nz. Use the existing product information as a base and enrich it by adding relevant, accurate, and product-specific content to the following columns:

- Description: Expand with at least 5 detailed, technical, and practical points (separate with semicolons). Include nutrient release profile, formulation type, compatibility with soil types, etc. Use both resupply.com and ravensdown.co.nz for additional information.
- Benefits: Add at least 5 practical, agronomic benefits (separate with semicolons). Use both resupply.com and ravensdown.co.nz for additional information.
- Application & Advice: Add at least 5 points on use cases, ideal crops/seasons, rates (with units), region-specific advice (separate with semicolons). Use both resupply.com and ravensdown.co.nz for additional information.
- Storage: Add at least 5 points on best practices, shelf life, packaging (separate with semicolons). Use both resupply.com and ravensdown.co.nz for additional information.
- Safety, Mixing and Compatibility: Add at least 5 points on safe handling, mixing do's/don'ts, incompatibilities (separate with semicolons). Use both resupply.com and ravensdown.co.nz for additional information.
- Nutrient_N, Nutrient_P, Nutrient_K, Nutrient_S, Nutrient_Ca, Nutrient_Mg: If ALL values are missing or 'NA', extract the correct values ONLY from https://www.resupply.co.nz for this product. If ANY value is present, set all missing or 'NA' values to 0. Use numeric values only, units as %.

Do NOT modify Name, Price, Unit, or URL. Only enrich the specified columns. Do not add new columns. Return the result as a JSON object with only the columns to be enriched or filled (Description, Benefits, Application & Advice, Storage, Safety, Mixing and Compatibility, Nutrient_N, Nutrient_P, Nutrient_K, Nutrient_S, Nutrient_Ca, Nutrient_Mg).

Here is the product data to enrich:
Name: {Name}
Price: {Price}
Unit: {Unit}
URL: {URL}
Description: {Description}
Benefits: {Benefits}
Application & Advice: {Application & Advice}
Storage: {Storage}
Safety, Mixing and Compatibility: {Safety, Mixing and Compatibility}
Nutrient_N: {Nutrient_N}
Nutrient_P: {Nutrient_P}
Nutrient_K: {Nutrient_K}
Nutrient_S: {Nutrient_S}
Nutrient_Ca: {Nutrient_Ca}
Nutrient_Mg: {Nutrient_Mg}
'''

def build_prompt(row):
    data = {col: str(row.get(col, "")) for col in COLUMNS}
    return PROMPT_TEMPLATE.format(**data)

def enrich_batch(rows):
    prompts = [build_prompt(row) for _, row in rows.iterrows()]
    responses = []
    for prompt in prompts:
        try:
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5000,
                temperature=0.3
            )
            enriched = response.choices[0].message.content
        except Exception as e:
            enriched = f"ERROR: {e}"
        responses.append(enriched)
        time.sleep(1.5)  # To avoid rate limits
    return responses

def main():
    df = pd.read_excel(INPUT_XLSX)
    if 'Enriched_Info' in df.columns:
        df = df.drop(columns=['Enriched_Info'])
    for col in ENRICH_COLUMNS + NUTRIENT_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    for i in range(0, len(df), BATCH_SIZE):
        batch = df.iloc[i:i+BATCH_SIZE].copy()
        # Preprocess nutrient columns as per new logic
        for idx, row in batch.iterrows():
            nutrients = [str(row[nc]).strip().upper() for nc in NUTRIENT_COLUMNS]
            all_na = all(n in ["", "NA"] for n in nutrients)
            any_present = any(n not in ["", "NA"] for n in nutrients)
            if any_present:
                # Set all NA or empty nutrient values to 0
                for nc in NUTRIENT_COLUMNS:
                    if str(row[nc]).strip().upper() in ["", "NA"]:
                        batch.at[idx, nc] = "0"
        batch_enriched = enrich_batch(batch)
        for j, enriched_json in enumerate(batch_enriched):
            try:
                enriched_data = pd.read_json('[' + enriched_json + ']', typ='series')[0] if enriched_json.strip().startswith('{') else {}
            except Exception:
                enriched_data = {}
            for col in ENRICH_COLUMNS + NUTRIENT_COLUMNS:
                if col in enriched_data and enriched_data[col]:
                    df.at[batch.index[j], col] = enriched_data[col]
        print(f"Processed {i + len(batch)} / {len(df)} products...")
    df.to_excel(OUTPUT_XLSX, index=False)
    print(f"✅ Enriched data saved to {OUTPUT_XLSX}")

if __name__ == "__main__":
    main() 