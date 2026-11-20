from pathlib import Path
import csv
import fitz
import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
PDF_PATH = ROOT / "data" / "NC-Standard-1.pdf"
PRICING_CSV = ROOT / "data" / "pricing_page1.csv"
DB_PATH = ROOT / "chroma_db"
COLLECTION_NAME = "cci_pricing_page1"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def extract_first_page():
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}\nPut NC-Standard-1.pdf inside the data folder."
        )
    with fitz.open(PDF_PATH) as pdf:
        if len(pdf) == 0:
            raise ValueError("The PDF has no pages.")
        return pdf[0].get_text("text")

def load_pricing_records():
    with PRICING_CSV.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))

def build_documents(pdf_text, pricing_records):
    docs = []
    metadatas = []
    ids = []

    # Short, topic-focused chunks based on the visible first page.
    info_chunks = [
        "Document: Standard Certified 150/35. Page 1. Product range: 12 to 24 feet wide, up to 16 feet high, any length subject to the pricing and roof-style conditions shown on this page.",
        "Page 1 general features: 29 gauge roofing, anchors included, and 5 foot centers. Four foot braces are shown on every leg for buildings 8 feet or taller.",
        "Page 1 height information: Standard heights are 6 feet to 16 feet. The diagram notes that 15 to 16 foot height requires a telescopic forklift.",
        "Page 1 roof-style condition: Buildings over 35 feet long must be vertical roof style.",
        "Page 1 installation condition: Land must be level prior to installation of the building.",
        "Page 1 vertical roof note: Roof length is 1 foot longer than frame length.",
        "Page 1 pricing table contains Vertical, Boxed Eave, and Regular roof styles. Dimensions are shown as width by length in feet. Prices are in US dollars.",
        "Page 1 source note: Prices are subject to change without notice. This chatbot reports only the values printed in the supplied PDF and does not confirm current prices."
    ]
    for i, chunk in enumerate(info_chunks, start=1):
        ids.append(f"info_{i}")
        docs.append(chunk)
        metadatas.append({"page": 1, "kind": "general", "source": "NC-Standard-1.pdf"})

    # One independently searchable document per exact price entry.
    for row in pricing_records:
        roof = row["roof_type"]
        width = int(row["width_ft"])
        length = int(row["length_ft"])
        price = int(row["price_usd"])
        doc_id = f"price_{roof.lower().replace(' ', '_')}_{width}x{length}"
        doc = (
            f"CCI pricing PDF page 1. Roof style: {roof}. "
            f"Building size: {width} feet wide by {length} feet long. "
            f"Listed price: ${price:,} USD."
        )
        ids.append(doc_id)
        docs.append(doc)
        metadatas.append({
            "page": 1,
            "kind": "price",
            "roof_type": roof,
            "width_ft": width,
            "length_ft": length,
            "price_usd": price,
            "source": "NC-Standard-1.pdf"
        })

    return ids, docs, metadatas

def main():
    print("Reading page 1 of the CCI PDF...")
    extracted_text = extract_first_page()
    (ROOT / "page1_extracted_text.txt").write_text(extracted_text, encoding="utf-8")
    print("Saved extracted text to page1_extracted_text.txt")

    pricing_records = load_pricing_records()
    ids, docs, metadatas = build_documents(extracted_text, pricing_records)

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(docs, show_progress_bar=True).tolist()

    client = chromadb.PersistentClient(path=str(DB_PATH))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # Add in batches to avoid oversized requests.
    batch_size = 100
    for start in range(0, len(docs), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            documents=docs[start:end],
            metadatas=metadatas[start:end],
            embeddings=embeddings[start:end]
        )

    print(f"\nIngestion complete. Stored {collection.count()} records in ChromaDB.")
    print("Pricing records loaded:", len(pricing_records))
    print("Only page 1 was used.")

if __name__ == "__main__":
    main()
