import json
import time
import os
from dotenv import load_dotenv
import requests
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters

# --- load secrets from .env (never hardcode these) ---
load_dotenv()
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
AIPIPE_TOKEN = os.environ["AIPIPE_TOKEN"]
# Fill this in once you've hosted run.jsonl publicly (Step 5 of the guide).
# For now a placeholder is fine while you're testing locally.
LOG_URL = os.environ.get("LOG_URL", "PASTE_YOUR_PUBLIC_LOG_URL_HERE")
# -------------------------------------------------------

GEMINI_URL = "https://aipipe.org/geminiv1beta/models/gemini-2.5-flash:generateContent"
LOG_FILE = "run.jsonl"

# Keeps the last few messages per chat, so multi-turn questions work —
# "answer the LAST message" still needs the earlier ones for context.
conversation_history = {}

SYSTEM_PROMPT = (
    "You are a careful data analyst. The user's LAST message asks a data-analysis "
    "question and tells you exactly what JSON shape to reply with. Work out the "
    "real answer (use any public data you know, e.g. MOSPI statistics, general "
    "world knowledge, or arithmetic on numbers given in the message). "
    "Reply with ONLY that exact JSON object and absolutely nothing else — no "
    "explanation, no markdown, no code fences, just the raw JSON."
)


def log_event(event: dict):
    event["timestamp"] = time.time()
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(event) + "\n")


def ask_gemini(history: list) -> str:
    """Send the conversation to Gemini (via AI Pipe) and return its raw text reply."""
    contents = [
        {"role": turn["role"], "parts": [{"text": turn["content"]}]}
        for turn in history
    ]
    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": contents,
    }
    r = requests.post(
        GEMINI_URL,
        headers={
            "Authorization": f"Bearer {AIPIPE_TOKEN}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_text = update.message.text
    log_event({"type": "incoming", "chat_id": chat_id, "text": user_text})

    # Gemini expects roles "user" and "model" (not "assistant")
    history = conversation_history.setdefault(chat_id, [])
    history.append({"role": "user", "content": user_text})

    reply_text = ask_gemini(history[-6:])
    history.append({"role": "model", "content": reply_text})

    # Make sure we actually reply with valid JSON containing "log_url" — if the model
    # added stray text or markdown fences, extract just the {...} part.
    try:
        parsed = json.loads(reply_text)
    except json.JSONDecodeError:
        start, end = reply_text.find("{"), reply_text.rfind("}")
        parsed = json.loads(reply_text[start:end + 1])

    parsed["log_url"] = LOG_URL
    final_reply = json.dumps(parsed)

    log_event({"type": "outgoing", "chat_id": chat_id, "text": final_reply})
    await update.message.reply_text(final_reply)


def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("Bot is running... press Ctrl+C to stop.")
    app.run_polling()


if __name__ == "__main__":
    main()