import json
import time
import os
import io
import subprocess
from dotenv import load_dotenv
import requests
import pandas as pd
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
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")  # only needed for auto-push on a host like Railway


def setup_git_auth():
    """Configure git identity + an authenticated remote, so push works with no
    interactive login (needed on a fresh host like Railway, unlike your own
    laptop where git already has cached credentials).

    Safe to call even if GITHUB_TOKEN isn't set (e.g. running locally) — it
    just skips silently and your normal local git setup is used instead.
    """
    if not GITHUB_TOKEN:
        return
    try:
        # Parse "owner/repo" out of LOG_URL, e.g.
        # https://raw.githubusercontent.com/kumarm-ds/tds-p1-q5/refs/heads/main/run.jsonl
        parts = LOG_URL.split("raw.githubusercontent.com/")[1].split("/")
        owner, repo = parts[0], parts[1]
        remote_url = f"https://x-access-token:{GITHUB_TOKEN}@github.com/{owner}/{repo}.git"

        subprocess.run(["git", "config", "user.email", "bot@example.com"], check=True)
        subprocess.run(["git", "config", "user.name", "Data Analyst Bot"], check=True)
        subprocess.run(["git", "remote", "set-url", "origin", remote_url], check=True)
        print("Git auth configured for auto-push.")
    except Exception as e:
        print(f"Could not configure git auth (auto-push will fail silently): {e}")

# Keeps the last few messages per chat, so multi-turn questions work —
# "answer the LAST message" still needs the earlier ones for context.
conversation_history = {}

SYSTEM_PROMPT = (
    "You are a careful data analyst. The user's LAST message asks a data-analysis "
    "question and tells you exactly what JSON shape to reply with. Work out the "
    "real answer using any data provided to you in this conversation (including any "
    "dataset content given to you), general world knowledge, or arithmetic on numbers "
    "given in the message. "
    "Reply with ONLY that exact JSON object and absolutely nothing else — no "
    "explanation, no markdown, no code fences, just the raw JSON."
)

PLAN_SYSTEM_PROMPT = (
    "You are a planning assistant for a data-analysis agent. Given a question, decide "
    "whether answering it accurately requires fetching an external public dataset "
    "(e.g. from MOSPI, data.gov.in, or a URL mentioned in the question), rather than "
    "relying on general knowledge or arithmetic on numbers already given inline. "
    "Reply with ONLY this JSON and nothing else: "
    '{"need_data": true or false, "url": "<direct CSV or XLSX download URL, or null>"}. '
    "Set need_data to false for general-knowledge, arithmetic, or inline-data questions. "
    "Only set a url if you are confident it is a real, direct, downloadable file link — "
    "never guess a plausible-looking URL you are not confident exists."
)


def log_event(event: dict):
    event["timestamp"] = time.time()
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(event) + "\n")


def push_log_to_git():
    """Commit and push run.jsonl so the public log_url always reflects the latest runs.

    Safe to call often, and safe to fail: if git isn't installed, there's nothing
    new to commit, or the push fails for any reason, we just log it and move on
    rather than ever crashing the bot or blocking a reply.
    """
    try:
        subprocess.run(["git", "add", LOG_FILE], check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", f"log update {time.time()}"],
            check=True,
            capture_output=True,
        )
        subprocess.run(["git", "push"], check=True, capture_output=True)
        print("Log pushed to GitHub.")
    except Exception as e:
        # Covers: nothing to commit, git not installed (FileNotFoundError), push
        # auth failure, network issues, etc. — all harmless to the bot's reply.
        print(f"Git push skipped/failed (often harmless): {e}")


def call_gemini(system_prompt: str, contents: list) -> str:
    """Low-level call to Gemini via AI Pipe. contents is a list of {role, parts} dicts."""
    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
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


def extract_json(raw: str) -> dict:
    """Parse JSON out of a model reply, tolerating stray text/markdown around it."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        return json.loads(raw[start:end + 1])


def plan_data_need(question: str) -> dict:
    """Ask Gemini whether this question needs an external dataset, and what URL."""
    raw = call_gemini(
        PLAN_SYSTEM_PROMPT,
        [{"role": "user", "parts": [{"text": question}]}],
    )
    return extract_json(raw)


EXPRESSION_SYSTEM_PROMPT = (
    "You will be given a question and the columns of a pandas DataFrame called `df`. "
    "Reply with ONLY a single valid Python expression (no explanation, no markdown, "
    "no assignment, no print) that computes the answer using `df` and pandas. "
    "Examples: df['total_bill'].mean()  |  df.groupby('state')['rate'].mean().idxmax() "
    "|  df['value'].max()"
)


def fetch_dataframe(url: str) -> pd.DataFrame:
    """Download a dataset (CSV or Excel) and load it into a pandas DataFrame."""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    buffer = io.BytesIO(resp.content)

    if url.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(buffer)
    try:
        return pd.read_csv(buffer)
    except Exception:
        buffer.seek(0)
        return pd.read_excel(buffer)


def summarize_df(df: pd.DataFrame) -> str:
    """Fallback text summary, used only if expression-based computation fails."""
    summary = f"Columns: {list(df.columns)}\nShape: {df.shape}\n\n"
    if len(df) <= 500:
        summary += f"Full data (CSV):\n{df.to_csv(index=False)}"
    else:
        summary += f"First 50 rows (CSV):\n{df.head(50).to_csv(index=False)}\n"
        summary += f"Summary statistics:\n{df.describe(include='all').to_csv()}"
    return summary


def get_pandas_expression(question: str, columns: list) -> str:
    """Ask Gemini for a pandas expression (not a final answer) to compute the result."""
    prompt = f"Question: {question}\nDataFrame columns: {columns}"
    raw = call_gemini(
        EXPRESSION_SYSTEM_PROMPT,
        [{"role": "user", "parts": [{"text": prompt}]}],
    )
    # Strip stray markdown fences/backticks the model might add anyway.
    expr = raw.strip().strip("`").strip()
    if expr.lower().startswith("python"):
        expr = expr[len("python"):].strip()
    return expr


def run_pandas_expression(expr: str, df: pd.DataFrame):
    """Execute a pandas expression against df in a restricted namespace.

    Not a perfect sandbox, but blocks builtins/imports so a stray expression
    can't do much beyond operate on df — good enough for a course assignment
    where the LLM itself generated the expression, not an untrusted user.
    """
    allowed_globals = {"__builtins__": {}}
    allowed_locals = {"df": df, "pd": pd}
    return eval(expr, allowed_globals, allowed_locals)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_text = update.message.text
    log_event({"type": "incoming", "chat_id": chat_id, "text": user_text})

    # Gemini expects roles "user" and "model" (not "assistant")
    history = conversation_history.setdefault(chat_id, [])
    history.append({"role": "user", "content": user_text})

    # --- Layer 2: figure out if this question needs a real external dataset ---
    data_context = ""
    try:
        plan = plan_data_need(user_text)
        log_event({"type": "plan", "chat_id": chat_id, "plan": plan})
        if plan.get("need_data") and plan.get("url"):
            df = fetch_dataframe(plan["url"])
            log_event({"type": "data_fetched", "chat_id": chat_id, "url": plan["url"]})

            try:
                expr = get_pandas_expression(user_text, list(df.columns))
                result = run_pandas_expression(expr, df)
                log_event({
                    "type": "computed",
                    "chat_id": chat_id,
                    "expression": expr,
                    "result": str(result),
                })
                data_context = (
                    f"This exact result was already computed for you directly from the "
                    f"real dataset using the pandas expression `{expr}`:\n"
                    f"Result: {result}\n"
                    f"Use this exact value as your answer — do not recompute or estimate it."
                )
            except Exception as e:
                # Expression failed (bad syntax, wrong column name, etc.) — fall back
                # to giving Gemini the raw data so it can still attempt an answer.
                log_event({"type": "compute_error", "chat_id": chat_id, "error": str(e)})
                data_context = summarize_df(df)
    except Exception as e:
        # If planning or fetching fails entirely, fall back to plain LLM reasoning
        # rather than crashing the bot or missing the reply window.
        log_event({"type": "data_fetch_error", "chat_id": chat_id, "error": str(e)})

    contents = [
        {"role": turn["role"], "parts": [{"text": turn["content"]}]}
        for turn in history[-6:]
    ]
    if data_context:
        contents.append({
            "role": "user",
            "parts": [{"text": f"Here is the relevant dataset to use for your answer:\n{data_context}"}],
        })

    reply_text = call_gemini(SYSTEM_PROMPT, contents)
    history.append({"role": "model", "content": reply_text})

    # Make sure we actually reply with valid JSON containing "log_url" — if the model
    # added stray text or markdown fences, extract just the {...} part.
    parsed = extract_json(reply_text)
    parsed["log_url"] = LOG_URL
    final_reply = json.dumps(parsed)

    log_event({"type": "outgoing", "chat_id": chat_id, "text": final_reply})
    await update.message.reply_text(final_reply)
    push_log_to_git()


def main():
    setup_git_auth()
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("Bot is running... press Ctrl+C to stop.")
    app.run_polling()


if __name__ == "__main__":
    main()