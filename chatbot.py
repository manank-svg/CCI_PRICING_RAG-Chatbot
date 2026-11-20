from pathlib import Path
import csv
import os
import re

import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "chroma_db"
PRICING_CSV = ROOT / "data" / "pricing_page1.csv"
COLLECTION_NAME = "cci_pricing_page1"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NO_INFO = "Sorry, this information is not available in the provided CCI pricing PDF (page 1)."

load_dotenv(ROOT / ".env")
API_KEY = os.getenv("GROQ_API_KEY", "").strip()
MODEL_NAME = os.getenv("GROQ_MODEL", "").strip()

def load_prices():
    with PRICING_CSV.open("r", newline="", encoding="utf-8") as file:
        records = list(csv.DictReader(file))
    for row in records:
        row["width_ft"] = int(row["width_ft"])
        row["length_ft"] = int(row["length_ft"])
        row["price_usd"] = int(row["price_usd"])
    return records

def parse_dimensions(question):
    # Accept forms such as 12 x 20, 12X20, 12 by 20, or 12' x 20'.
    match = re.search(
        r"(?<!\d)(\d{1,2})\s*(?:ft|feet|')?\s*(?:x|×|by)\s*(\d{1,2})\s*(?:ft|feet|')?(?!\d)",
        question,
        flags=re.IGNORECASE
    )
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))

def detect_roof_type(question):
    q = question.lower()
    found = []
    if re.search(r"\bvertical\b", q):
        found.append("Vertical")
    if re.search(r"\bbox(?:ed)?[\s-]*eave\b", q):
        found.append("Boxed Eave")
    if re.search(r"\bregular\b", q):
        found.append("Regular")
    if len(found) == 1:
        return found[0]
    return None

def is_price_question(question):
    q = question.lower()
    return (
        parse_dimensions(question) is not None
        or any(word in q for word in ["price", "cost", "how much", "pricing", "dollar", "$"])
    )

def exact_price_answer(question, records):
    dimensions = parse_dimensions(question)
    roof = detect_roof_type(question)

    if dimensions is None:
        return "Please specify the building dimensions in width x length format, for example 12 x 20, and the roof style."
    width, length = dimensions

    if roof is None:
        return "Please specify the roof style: Vertical, Boxed Eave, or Regular. I need it to avoid giving you a price from the wrong pricing column."

    if length > 35 and roof != "Vertical":
        return (
            "The PDF states that buildings over 35 feet long must be vertical roof style. "
            "Please ask for the Vertical price for this size."
        )

    for row in records:
        if (
            row["roof_type"] == roof
            and row["width_ft"] == width
            and row["length_ft"] == length
        ):
            return (
                f"the {roof} building measuring {width} ft wide x {length} ft long "
                f"is listed at ${row['price_usd']:,} USD. "
            )

    return NO_INFO

def retrieve_context(question, collection, embedding_model, top_k=5):
    query_embedding = embedding_model.encode([question]).tolist()
    result = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    # Conservative relevance filter for cosine distance.
    selected = []
    for doc, metadata, distance in zip(documents, metadatas, distances):
        if distance <= 0.55:
            selected.append(
                f"[Source: page {metadata.get('page', 1)}; "
                f"section: {metadata.get('kind', 'general')}]\n{doc}"
            )
    return "\n\n".join(selected)

def ask_llm(question, context, groq_client):
    if not context.strip():
        return NO_INFO

    system_prompt = """
You are a careful assistant answering questions using ONLY the supplied
CCI pricing PDF page 1 excerpts.

Rules:
- Use only facts explicitly present in the supplied context.
- Never invent, estimate, calculate, or assume a price.
- Never mix roof styles, dimensions, or pricing rows.
- If the requested fact is not explicitly supported by the context, say:
  "Sorry, this information is not available in the provided CCI pricing PDF (page 1)."
- Do not use outside knowledge.
- Mention that the source is page 1 when appropriate.
- Be concise.
"""

    response = groq_client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Retrieved PDF context:\n{context}\n\n"
                    f"Question: {question}\n\n"
                    "Answer only from the retrieved context."
                )
            }
        ]
    )
    return response.choices[0].message.content.strip()

def main():
    if not DB_PATH.exists():
        print("Vector database not found. First run: python ingest.py")
        return
    if not API_KEY:
        print("GROQ_API_KEY is missing. Add it to the .env file.")
        return
    if not MODEL_NAME:
        print("GROQ_MODEL is missing. Add a model available to your Groq account in .env.")
        return

    client = chromadb.PersistentClient(path=str(DB_PATH))
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        print("ChromaDB collection not found. First run: python ingest.py")
        return

    print("Loading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    groq_client = Groq(api_key=API_KEY)
    pricing_records = load_prices()

    print("\nCCI Pricing RAG Chatbot (PDF page 1)")
    print("Ask about prices, roof styles, dimensions, or specifications.")
    print("Type 'exit' to stop.\n")

    # Store previous question details
    pending_dimensions = None
    while True:
        question = input("You: ").strip()
        current_dimensions = parse_dimensions(question)
        current_roof = detect_roof_type(question)
        # Check whether the bot is waiting for a roof style
        if pending_dimensions is not None:
            # Combine previous dimensions with the new roof style
            if current_roof is not None:
                width, length = pending_dimensions
                answer = exact_price_answer(
                    f"{width} x {length} {current_roof}",
                    pricing_records
                )
                print("Bot:", answer)
                pending_dimensions = None  # Reset after answering
                continue
            else:
                print("Bot: Please specify the roof style: Vertical, Boxed Eave, or Regular.")
                continue

        if question.lower() in {"exit", "quit"}:
            print("Bot: Goodbye!")
            break
        if not question:
            print("Bot: Please enter a question.")
            continue

        # Exact pricing is answered from a verified structured row, not guessed by the LLM.
        if is_price_question(question):

            dimensions = parse_dimensions(question)
            roof = detect_roof_type(question)

            if dimensions is None:
                print("Bot: Please provide building dimensions.")
                continue

            if roof is None:

               # Remember dimensions for the next user message
                pending_dimensions = dimensions

                print(
                    f"Bot: Which roof style do you want "
                    f"for {dimensions[0]} x {dimensions[1]} ft?"
                )

                continue

            answer = exact_price_answer(
                f"{dimensions[0]} x {dimensions[1]} {roof}",
                pricing_records
            )

            print("Bot:", answer)

            continue

        context = retrieve_context(question, collection, embedding_model)
        answer = ask_llm(question, context, groq_client)
        print("Bot:", answer)

if __name__ == "__main__":
    main()
