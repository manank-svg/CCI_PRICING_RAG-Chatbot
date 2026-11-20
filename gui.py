"""PySide6 desktop GUI for the CCI Pricing RAG chatbot."""
from __future__ import annotations

import html
import os
import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from groq import Groq
from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtGui import QFont, QTextCursor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QPlainTextEdit, QPushButton, QSizePolicy, QTextBrowser, QVBoxLayout,
    QWidget,
)
from sentence_transformers import SentenceTransformer

import chatbot as core

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


class ChatWorker(QObject):
    """Runs model loading and all RAG/API work away from the GUI thread."""
    ready = Signal(str)
    answer_ready = Signal(str)
    error = Signal(str)

    def __init__(self):
        super().__init__()
        self.collection = None
        self.embedding_model = None
        self.groq_client = None
        self.pricing_records = []
        self.pending_dimensions = None
        self.history = []

    @Slot()
    def initialize(self):
        try:
            if not core.DB_PATH.exists():
                raise RuntimeError("Vector database not found. Run: python ingest.py")
            api_key = os.getenv("GROQ_API_KEY", "").strip()
            model_name = os.getenv("GROQ_MODEL", "").strip()
            if not api_key or api_key == "put_your_groq_api_key_here":
                raise RuntimeError("GROQ_API_KEY is missing. Add your key to the .env file.")
            if not model_name:
                raise RuntimeError("GROQ_MODEL is missing. Add a model supported by your Groq account to .env.")

            db_client = chromadb.PersistentClient(path=str(core.DB_PATH))
            self.collection = db_client.get_collection(core.COLLECTION_NAME)
            self.ready.emit("Loading embedding model…")
            self.embedding_model = SentenceTransformer(core.EMBEDDING_MODEL)
            self.groq_client = Groq(api_key=api_key)
            self.pricing_records = core.load_prices()
            self.ready.emit("Ready · CCI Pricing Assistant")
        except Exception as exc:
            self.error.emit(str(exc))

    @Slot(str)
    def process_question(self, question: str):
        try:
            if self.collection is None or self.embedding_model is None:
                raise RuntimeError("The chatbot is not initialized. Check the setup message and restart the app.")

            q = question.strip()
            low = q.lower()
            if low in {"cancel", "never mind", "nevermind"}:
                self.pending_dimensions = None
                self.answer_ready.emit("No problem. I cancelled the pending price question. What would you like to know?")
                return
            if low in {"reset", "clear", "start over"}:
                self.pending_dimensions = None
                self.history.clear()
                self.answer_ready.emit("Conversation context cleared. How can I help you?")
                return

            current_roof = core.detect_roof_type(q)
            if self.pending_dimensions is not None:
                if current_roof is not None:
                    width, length = self.pending_dimensions
                    answer = core.exact_price_answer(f"{width} x {length} {current_roof}", self.pricing_records)
                    self.pending_dimensions = None
                    self.history.extend([(q, answer)])
                    self.answer_ready.emit(answer)
                    return
                # If the user supplies a new size, treat it as a new price request.
                if core.parse_dimensions(q) is None:
                    self.answer_ready.emit("I'm still waiting for the roof style for your previous size. Please reply Vertical, Boxed Eave, or Regular — or type cancel.")
                    return
                self.pending_dimensions = None

            if core.is_price_question(q):
                dimensions = core.parse_dimensions(q)
                roof = current_roof
                if dimensions is None:
                    answer = "Please provide building dimensions in width x length format, for example 22 x 20, and the roof style."
                elif roof is None:
                    self.pending_dimensions = dimensions
                    answer = f"What roof style do you want for {dimensions[0]} × {dimensions[1]} ft? Please choose Vertical, Boxed Eave, or Regular."
                else:
                    answer = core.exact_price_answer(f"{dimensions[0]} x {dimensions[1]} {roof}", self.pricing_records)
                self.history.append((q, answer))
                self.history = self.history[-8:]
                self.answer_ready.emit(answer)
                return

            # Include recent turns in retrieval so follow-up questions have context.
            recent = self.history[-6:]
            contextual_query = "\n".join([f"{role}: {text}" for pair in recent for role, text in [("User", pair[0]), ("Assistant", pair[1])]] + [f"User: {q}"])
            context = core.retrieve_context(contextual_query, self.collection, self.embedding_model)
            if not context.strip():
                answer = core.NO_INFO
            else:
                system_prompt = (
                    "You are a careful assistant answering questions using ONLY the supplied CCI pricing PDF page 1 excerpts. "
                    "Use only facts explicitly present in the supplied context. Never invent or assume facts or prices. "
                    "If the requested fact is not supported, say: " + core.NO_INFO +
                    " Do not use outside knowledge. Be concise. Use the conversation history to resolve follow-up references."
                )
                messages = [{"role": "system", "content": system_prompt}]
                for old_q, old_a in recent:
                    messages.append({"role": "user", "content": old_q})
                    messages.append({"role": "assistant", "content": old_a})
                messages.append({"role": "user", "content": f"Retrieved PDF context:\n{context}\n\nQuestion: {q}\nAnswer only from the retrieved context."})
                result = self.groq_client.chat.completions.create(
                    model=os.getenv("GROQ_MODEL", "").strip(),
                    temperature=0,
                    messages=messages,
                )
                answer = result.choices[0].message.content.strip()
            self.history.append((q, answer))
            self.history = self.history[-8:]
            self.answer_ready.emit(answer)
        except Exception as exc:
            self.error.emit(str(exc))

    @Slot()
    def reset_conversation(self):
        self.pending_dimensions = None
        self.history.clear()


class ChatWindow(QMainWindow):
    ask_requested = Signal(str)
    reset_requested = Signal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("CCI Pricing Assistant")
        self.resize(1050, 760)
        self.setMinimumSize(720, 560)
        self._build_ui()

        self.thread = QThread(self)
        self.worker = ChatWorker()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.initialize)
        self.ask_requested.connect(self.worker.process_question)
        self.reset_requested.connect(self.worker.reset_conversation)
        self.worker.ready.connect(self.on_ready)
        self.worker.answer_ready.connect(self.on_answer)
        self.worker.error.connect(self.on_error)
        self.thread.start()

    def _build_ui(self):
        central = QWidget()
        central.setObjectName("central")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(28, 24, 28, 22)
        root.setSpacing(16)

        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("CCI Pricing Assistant")
        title.setObjectName("title")
        subtitle = QLabel("Ask about building prices, roof styles and specifications")
        subtitle.setObjectName("subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box)
        header.addStretch(1)
        self.status = QLabel("Starting…")
        self.status.setObjectName("status")
        header.addWidget(self.status, alignment=Qt.AlignVCenter)
        root.addLayout(header)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setObjectName("separator")
        root.addWidget(line)

        self.chat = QTextBrowser()
        self.chat.setObjectName("chat")
        self.chat.setOpenExternalLinks(False)
        self.chat.setReadOnly(True)
        self.chat.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root.addWidget(self.chat, 1)
        self._append_message("assistant", "Hello! I'm your CCI Pricing Assistant. Ask me about building prices or specifications from the supplied PDF.\n\nFor a price, you can ask something like: **What is the price of a 22 x 20 building?** I'll ask for the roof style if you haven't provided it.")

        quick = QHBoxLayout()
        quick.addWidget(QLabel("Try:"))
        for label, prompt in [("Building price", "What is the price of a 22 x 20 building?"), ("Roof styles", "What roof styles are available?"), ("Building height", "What is the standard building height?")]:
            btn = QPushButton(label)
            btn.setObjectName("quickButton")
            btn.clicked.connect(lambda checked=False, p=prompt: self._send_text(p))
            quick.addWidget(btn)
        quick.addStretch(1)
        root.addLayout(quick)

        composer = QHBoxLayout()
        composer.setSpacing(10)
        self.input = QPlainTextEdit()
        self.input.setObjectName("input")
        self.input.setPlaceholderText("Type your message…  (Enter to send, Shift+Enter for a new line)")
        self.input.setFixedHeight(72)
        self.input.setTabChangesFocus(True)
        composer.addWidget(self.input, 1)
        self.send_button = QPushButton("Send  ➤")
        self.send_button.setObjectName("sendButton")
        self.send_button.setFixedSize(112, 72)
        self.send_button.clicked.connect(self.send_message)
        composer.addWidget(self.send_button)
        root.addLayout(composer)

        footer = QHBoxLayout()
        hint = QLabel("Answers are based on the provided CCI PDF page 1 and pricing data.")
        hint.setObjectName("hint")
        footer.addWidget(hint)
        footer.addStretch(1)
        self.clear_button = QPushButton("Clear conversation")
        self.clear_button.setObjectName("clearButton")
        self.clear_button.clicked.connect(self.clear_conversation)
        footer.addWidget(self.clear_button)
        root.addLayout(footer)

        self.input.installEventFilter(self)
        QShortcut(QKeySequence("Ctrl+Return"), self, activated=self.send_message)
        self.setStyleSheet(STYLE)

    def eventFilter(self, obj, event):
        # Enter sends; Shift+Enter remains a newline in the text editor.
        from PySide6.QtCore import QEvent
        from PySide6.QtGui import QKeyEvent
        if obj is self.input and event.type() == QEvent.KeyPress:
            if event.key() in (Qt.Key_Return, Qt.Key_Enter) and not (event.modifiers() & Qt.ShiftModifier):
                self.send_message()
                return True
        return super().eventFilter(obj, event)

    def _append_message(self, role: str, text: str):
        safe = html.escape(text).replace("\n", "<br>")
        # Basic Markdown bold support for the welcome and answer text.
        import re
        safe = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe)
        if role == "user":
            block = f'''<div style="margin:14px 0 14px 18%; text-align:right;">
              <div style="color:#94a3b8;font-size:11px;margin:0 8px 5px;">YOU</div>
              <div style="display:inline-block;text-align:left;background:#2457d6;color:#ffffff;padding:13px 16px;border-radius:16px 16px 4px 16px;max-width:88%;line-height:1.55;">{safe}</div></div>'''
        else:
            block = f'''<div style="margin:14px 18% 14px 0;">
              <div style="color:#65a7ff;font-size:11px;margin:0 8px 5px;">CCI ASSISTANT</div>
              <div style="display:inline-block;background:#202b3b;color:#e7edf7;padding:13px 16px;border-radius:16px 16px 16px 4px;max-width:94%;line-height:1.6;">{safe}</div></div>'''
        self.chat.moveCursor(QTextCursor.End)
        self.chat.insertHtml(block)
        self.chat.insertHtml("<div style='height:3px'></div>")
        self.chat.moveCursor(QTextCursor.End)

    def _send_text(self, text):
        self.input.setPlainText(text)
        self.send_message()

    def send_message(self):
        if not self.send_button.isEnabled():
            return
        question = self.input.toPlainText().strip()
        if not question:
            return
        self._append_message("user", question)
        self.input.clear()
        self.send_button.setEnabled(False)
        self.send_button.setText("…")
        self.status.setText("Thinking…")
        self.ask_requested.emit(question)

    @Slot(str)
    def on_answer(self, answer):
        self._append_message("assistant", answer)
        self.send_button.setEnabled(True)
        self.send_button.setText("Send  ➤")
        self.status.setText("Ready · CCI Pricing Assistant")
        self.input.setFocus()

    @Slot(str)
    def on_ready(self, message):
        self.status.setText(message)

    @Slot(str)
    def on_error(self, message):
        self._append_message("assistant", f"I couldn't complete that request.\n\n{message}")
        self.send_button.setEnabled(True)
        self.send_button.setText("Send  ➤")
        self.status.setText("Needs attention · Check setup")

    def clear_conversation(self):
        self.chat.clear()
        self.reset_requested.emit()
        self._append_message("assistant", "Conversation cleared. Your RAG database is unchanged. What would you like to ask?")

    def closeEvent(self, event):
        self.thread.quit()
        self.thread.wait(5000)
        event.accept()


STYLE = """
QMainWindow, QWidget#central { background: #111827; color: #e5e7eb; }
QLabel#title { color: #f8fafc; font-size: 25px; font-weight: 700; }
QLabel#subtitle { color: #94a3b8; font-size: 12px; margin-top: 3px; }
QLabel#status { color: #93c5fd; background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 7px 12px; font-size: 11px; }
QFrame#separator { color: #293548; background: #293548; max-height: 1px; }
QTextBrowser#chat { background: #172131; color: #e5e7eb; border: 1px solid #293548; border-radius: 16px; padding: 14px; selection-background-color: #315fc4; }
QPlainTextEdit#input { background: #1e293b; color: #f8fafc; border: 1px solid #3b4a60; border-radius: 12px; padding: 12px; font-size: 13px; }
QPlainTextEdit#input:focus { border: 1px solid #4f83f1; }
QPushButton { border: none; border-radius: 9px; padding: 9px 13px; font-weight: 600; }
QPushButton#sendButton { background: #3167e8; color: white; font-size: 13px; }
QPushButton#sendButton:hover { background: #4278f5; }
QPushButton#sendButton:disabled { background: #475569; color: #cbd5e1; }
QPushButton#quickButton { background: #202d40; color: #cbd5e1; border: 1px solid #34445b; font-size: 11px; }
QPushButton#quickButton:hover { background: #2b3c55; }
QPushButton#clearButton { background: transparent; color: #93c5fd; border: 1px solid #34445b; font-size: 11px; }
QLabel#hint, QHBoxLayout QLabel { color: #8493a8; font-size: 11px; }
"""


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("CCI Pricing Assistant")
    app.setFont(QFont("Segoe UI", 10))
    window = ChatWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
