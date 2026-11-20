# CCI Pricing RAG Chatbot — PySide6 Desktop GUI

This version adds a PySide6 desktop interface to the uploaded RAG chatbot. It retains the existing `chatbot.py`, `ingest.py`, PDF, pricing CSV, and ChromaDB collection.

## 1. Open the project

Extract the ZIP and open the `CCI_RAG_Chatbot_PySide6` folder in VS Code.

## 2. Create and activate a virtual environment (Windows)

```powershell
python -m venv .venv
.venv\Scripts\activate
```

## 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Configure Groq

Copy `.env.example` to a new file named `.env`:

```powershell
copy .env.example .env
```

Open `.env` and set your own Groq API key and a model currently available to your Groq account:

```env
GROQ_API_KEY=your_actual_groq_api_key
GROQ_MODEL=your_supported_model_name
```

Do not share your `.env` file or API key.

## 5. Run the desktop app

The supplied `chroma_db` folder is included. If it is missing or you change the source PDF/pricing data, rebuild it with:

```powershell
python ingest.py
```

Then launch the GUI:

```powershell
python gui.py
```

Alternatively, after creating `.venv` and installing requirements, double-click `launch_gui.bat`.

## Features

- Desktop chat window with message bubbles.
- Quick prompts for common questions.
- Exact price lookup from `data/pricing_page1.csv`.
- Follow-up roof-style handling: dimensions are retained while the app waits for `Vertical`, `Boxed Eave`, or `Regular`.
- Conversation history is included when retrieving context for general follow-up questions.
- Clear conversation resets chat context but does not delete the ChromaDB database.
- Model loading and requests run in a background Qt thread so the window remains responsive.

## Notes

- The `.env` file is intentionally not included. Add your own key locally.
- The pricing CSV in the supplied project is manually transcribed. Verify every price against the original PDF before using it for commercial quotes.
- The chatbot is grounded in page 1 of the supplied PDF; it does not establish that listed prices are current.
