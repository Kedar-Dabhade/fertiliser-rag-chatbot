import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
index_name = "expanded-fert"  # or your actual index name

# Initialize Pinecone client and get/create index
pc = Pinecone(api_key=pinecone_api_key)
if not pc.list_indexes().names().__contains__(index_name):
    pc.create_index(
        name=index_name,
        dimension=1536,  # OpenAI embedding dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=pinecone_env)
    )
index = pc.Index(index_name)

# Set up embeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Products list: only 'name' and 'information' keys
products = [
    {
        "name": "Urea (20kg Bag)",
        "information": """
Price: $24.76 / bag
URL: https://resupply.co.nz/collections/fertiliser/products/urea-20kg-bag
Nutrients: N: 46.0, P: 0.0, K: 0.0, S: 0.0, Ca: 0.0, Mg: 0.0

Description:
Urea is a highly concentrated nitrogen fertiliser containing 46% nitrogen, making it the most nitrogen-rich solid fertiliser available. It is widely used across various agricultural, horticultural, and forestry systems to promote vigorous plant growth and increase biomass production. Urea is a white, crystalline substance that dissolves readily in water, allowing for flexible application methods, including broadcasting and foliar spraying. Its rapid nitrogen release supports quick plant response, making it ideal for correcting nitrogen deficiencies. Urea is suitable for a broad range of crops and soil types, providing a versatile solution for nitrogen fertilisation needs.

Benefits:
- Provides the highest nitrogen content of any solid fertiliser, offering a cost-effective nitrogen source.
- Enhances plant growth and development, leading to increased yields and improved crop quality.
- Versatile application methods, including soil incorporation and foliar spraying, cater to different farming practices.
- Rapid nutrient availability allows for quick correction of nitrogen deficiencies.
- Suitable for a wide range of crops, including cereals, pastures, and horticultural plants.

Application & Advice:
- Apply at rates between 40 and 300 kg/ha, depending on crop requirements and soil conditions.
- For optimal results, apply during periods of active plant growth to maximize nitrogen uptake.
- In summer, apply just before or during rain to reduce potential nitrogen losses through volatilisation.
- Ensure even application by calibrating spreading equipment properly to prevent uneven nutrient distribution.
- Regular soil testing is recommended to monitor nitrogen levels and adjust application rates accordingly.

Storage:
- Store in a cool, dry, well-ventilated area to prevent moisture absorption and caking.
- Keep bags sealed when not in use to maintain product quality.
- Avoid prolonged storage; use within the recommended shelf life for optimal effectiveness.
- Store away from incompatible substances and potential contaminants.
- Ensure storage area is secure and accessible only to authorized personnel.

Safety, Mixing and Compatibility:
- Urea is not compatible with Superphosphate, Sulphur Super 30, Magnesium Oxide, Calcium Ammonium Nitrate, Copper Sulphate, and Cobalt Sulphate.
- Flexi-N (magnesium-coated Urea) can be used as a substitute when mixing with superphosphate-based products.
- Incompatible with hypochlorites (e.g., bleach), nitrates, nitrites, and strong oxidisers.
- Segregation may occur when mixed with products of differing particle sizes; ensure uniformity for effective application.
- Use appropriate personal protective equipment (PPE) when handling to prevent skin and eye contact.
"""
    },
    {
        "name": "Cropmaster® 20",
        "information": """
Price: $1,015 / mt
URL: https://resupply.co.nz/collections/fertiliser/products/cropmaster%C2%AE-20
Nutrients: N: 18.8, P: 10.0, K: 0.0, S: 12.0, Ca: 0.0, Mg: 0.0

Description:
Cropmaster® 20 is a balanced fertiliser blend containing 18.8% nitrogen (N), 10% phosphorus (P), and 12% sulphur (S), designed to support the growth of various crops, including field crops, green feed brassicas, and pastures for sheep, beef, and dairy. The nitrogen is present in the ammonium form, providing readily available nutrients for plant uptake. This formulation is particularly suitable for cropping situations where potassium is not required. The granulated form ensures ease of application and uniform nutrient distribution. Cropmaster® 20 is ideal for maintenance fertilisation, promoting robust plant growth and development.

Benefits:
- Provides a balanced supply of essential nutrients, supporting comprehensive plant development.
- The ammonium form of nitrogen ensures immediate availability for plant uptake, enhancing growth.
- Suitable for a wide range of crops, including cereals, pastures, and brassicas.
- Ideal for situations where potassium supplementation is not necessary, simplifying nutrient management.
- Granulated form allows for easy handling and even application, ensuring uniform nutrient distribution.

Application & Advice:
- Apply at rates based on soil test results and specific crop requirements; consult with an agronomist for tailored advice.
- Ensure equipment is calibrated correctly to achieve even spreading and prevent nutrient imbalances.
- Avoid applying at rates that may lead to nutrient runoff or leaching; adhere to recommended guidelines.
- Suitable for use in maintenance fertilisation programs to sustain soil fertility and crop productivity.
- Regular soil testing is recommended to monitor nutrient levels and adjust application rates accordingly.

Storage:
- Store in a cool, dry, well-ventilated area to maintain product quality and prevent caking.
- Keep bags sealed when not in use to protect against moisture ingress.
- Avoid stacking bags too high to prevent compaction and potential damage.
- Ensure storage area is free from contaminants that could compromise the fertiliser's integrity.
- Use older stock first to maintain product freshness and effectiveness.

Safety, Mixing and Compatibility:
- Cropmaster® 20 generally mixes well with other products but is not compatible with Superphosphate, Sulphur Super 30, and Magnesium Oxide.
- Segregation may occur when mixed with products of differing particle sizes; ensure uniformity for effective application.
- Always refer to the product compatibility chart before mixing to avoid adverse reactions.
- Use appropriate personal protective equipment (PPE) when handling to prevent skin and eye contact.
- Clean equipment thoroughly after use to prevent corrosion and contamination.
"""
    },
    {
        "name": "Cropmaster® 15",
        "information": """
Price: $1,084 / mt
URL: https://resupply.co.nz/collections/fertiliser/products/cropmaster%C2%AE-15
Nutrients: N: 14.8, P: 10.0, K: 10.0, S: 7.4, Ca: 0.0, Mg: 0.0

Description:
Cropmaster® 15 is a comprehensive fertiliser blend containing 14.8% nitrogen (N), 10% phosphorus (P), 10% potassium (K), and 7.4% sulphur (S), designed to meet the nutrient requirements of pastures, cereals, and maize. This formulation includes DAP, potassium chloride, and ammonium sulphate, providing a balanced nutrient supply for optimal plant growth. It is particularly suitable for cropping situations where potassium is required, supporting robust root development and overall plant health. The granulated form ensures ease of application and uniform nutrient distribution. Cropmaster® 15 is an economical choice for farmers seeking a versatile fertiliser solution.

Benefits:
- Delivers a balanced supply of essential nutrients, supporting comprehensive plant development.
- The inclusion of potassium supports root development and enhances drought resistance.
- Suitable for a wide range of crops, including pastures, cereals, and maize.
- Ideal for situations where potassium supplementation is necessary, simplifying nutrient management.
- Granulated form allows for easy handling and even application, ensuring uniform nutrient distribution.

Application & Advice:
- Apply at rates based on soil test results and specific crop requirements; consult with an agronomist for tailored advice.
- Ensure equipment is calibrated correctly to achieve even spreading and prevent nutrient imbalances.
- Avoid applying at rates that may lead to nutrient runoff or leaching; adhere to recommended guidelines.
- Suitable for use in maintenance fertilisation programs to sustain soil fertility and crop productivity.
- Regular soil testing is recommended to monitor nutrient levels and adjust application rates accordingly.

Storage:
- Store in a cool, dry, well-ventilated area to maintain product quality and prevent caking.
- Keep bags sealed when not in use to protect against moisture ingress.
- Avoid stacking bags too high to prevent compaction and potential damage.
- Ensure storage area is free from contaminants that could compromise the fertiliser's integrity.
- Use older stock first to maintain product freshness and effectiveness.

Safety, Mixing and Compatibility:
- Cropmaster® 15 generally mixes well with other products but is not compatible with Superphosphate, Sulphur Super 30, and Magnesium Oxide.
- Segregation may occur when mixed with products of differing particle sizes; ensure uniformity for effective application.
- Always refer to the product compatibility chart before mixing to avoid adverse reactions.
- Use appropriate personal protective equipment (PPE) when handling to prevent skin and eye contact.
- Clean equipment thoroughly after use to prevent corrosion and contamination.
"""
    },
    {
        "name": "Superphosphate (20kg Bag)",
        "information": """
Price: $15.63 / bag
URL: https://resupply.co.nz/collections/fertiliser/products/superphosphate-20kg-bag
Nutrients: N: 0.0, P: 9.0, K: 0.0, S: 11.0, Ca: 20.0, Mg: 0.0

Description:
Superphosphate is a granular fertiliser containing 9% phosphorus (P) and 11% sulphur (S), essential nutrients for optimal pasture growth. It is particularly effective in promoting root development and enhancing plant vigor. The formulation ensures rapid nutrient availability, making it suitable for immediate plant uptake. Superphosphate is compatible with a variety of soil types and is commonly used in both maintenance and capital fertiliser applications. Its consistent granule size allows for even spreading, ensuring uniform nutrient distribution across the field.

Benefits:
- Provides a cost-effective solution for supplying essential phosphorus and sulphur to pastures.
- Enhances root development, leading to improved plant establishment and growth.
- Suitable for regular maintenance applications as well as for developing new pasture areas.
- Rapid nutrient availability supports quick plant response and recovery.
- Compatible with various soil types, making it versatile for different farming systems.

Application & Advice:
- Recommended application rates vary based on soil tests and specific crop requirements; consult with an agronomist for tailored advice.
- Avoid applying at rates exceeding 1 tonne per hectare to prevent potential nutrient imbalances.
- Ideal for use in both maintenance fertilisation and during the establishment of new pastures.
- Ensure even spreading to achieve uniform nutrient distribution; calibrate spreading equipment accordingly.
- Soil testing is recommended to determine the appropriate application rate and to monitor soil nutrient levels over time.

Storage:
- Store in a cool, dry, well-ventilated area to prevent moisture absorption and caking.
- Keep bags sealed when not in use to maintain product quality.
- Avoid stacking bags too high to prevent compaction and potential damage.
- Ensure storage area is free from contaminants that could compromise the fertiliser's integrity.
- Use older stock first to maintain product freshness and effectiveness.

Safety, Mixing and Compatibility:
- Superphosphate generally mixes well with other products but is not compatible with Potassium Nitrate, Ammonium Nitrate, Mono Ammonium Phosphate (MAP), Cropmaster DAP, Calcium Ammonium Nitrate (CAN), or Urea.
- It can be compatible with Flexi-N (magnesium-coated Urea) and Ammonium Sulphate Granular under certain circumstances.
- Segregation may occur when mixed with products of differing particle sizes; ensure uniformity for effective application.
- Always refer to the product compatibility chart before mixing to avoid adverse reactions.
- Use appropriate personal protective equipment (PPE) when handling to prevent skin and eye contact.
"""
    },
    {
        "name": "N-Protect®",
        "information": """
Price: $1,003 / mt
URL: https://resupply.co.nz/collections/fertiliser/products/n-protect%C2%AE
Nutrients: N: 45.9, P: 0.0, K: 0.0, S: 0.0, Ca: 0.0, Mg: 0.0

Description:
N-Protect® is a coated urea fertiliser designed to reduce nitrogen losses through volatilisation by slowing the conversion of urea to ammonia gas. This ensures more nitrogen remains available for plant uptake, enhancing crop growth and reducing environmental impact. The coating contains 300 ppm of the active ingredient NBPT, a urease inhibitor that effectively minimizes nitrogen loss. N-Protect® is suitable for a wide range of crops and is particularly beneficial in conditions prone to nitrogen loss. Its use supports sustainable farming practices by improving nitrogen use efficiency.

Benefits:
- Reduces nitrogen losses through volatilisation by up to 50%, ensuring more efficient use of applied nitrogen.
- Enhances crop yield and quality by maintaining higher nitrogen availability in the root zone.
- Supports environmental sustainability by decreasing ammonia emissions into the atmosphere.
- Provides flexibility in application timing, allowing for fertilisation under a broader range of weather conditions.
- Compatible with various cropping systems, including cereals, pastures, and horticultural crops.

Application & Advice:
- Apply at rates between 40 and 300 kg/ha, depending on crop requirements and soil conditions.
- For optimal results, apply during periods of active plant growth to maximize nitrogen uptake.
- Avoid mixing with incompatible products; refer to compatibility guidelines before blending.
- Ensure even application by calibrating spreading equipment properly to prevent uneven nutrient distribution.
- Regular soil testing is recommended to monitor nitrogen levels and adjust application rates accordingly.

Storage:
- Store in a cool, dry, well-ventilated area to maintain product integrity.
- Keep bags sealed when not in use to prevent moisture ingress and caking.
- Avoid prolonged storage; use within the recommended shelf life for optimal effectiveness.
- Store away from incompatible substances and potential contaminants.
- Ensure storage area is secure and accessible only to authorized personnel.

Safety, Mixing and Compatibility:
- N-Protect® may be mixed with other products for prompt application but is not compatible with Superphosphate, Sulphur Super 30, Magnesium Oxide, Copper Sulphate, and Cobalt Sulphate.
- Flexi-N (magnesium-coated Urea) can be used as a substitute when mixing with superphosphate-based products.
- Incompatible with hypochlorites (e.g., bleach), nitrates, nitrites, and strong oxidisers.
- Segregation may occur when mixed with products of differing particle sizes; ensure uniformity for effective application.
- Use appropriate personal protective equipment (PPE) when handling to prevent skin and eye contact.
"""
    }
]

docs = []
for prod in products:
    doc = Document(
        page_content=prod["information"],
        metadata={
            "name": prod["name"]
        }
    )
    docs.append(doc)

# Create the vector store and add documents
vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
vectorstore.add_documents(docs)

print("✅ Successfully embedded and uploaded full product docs to Pinecone.") 