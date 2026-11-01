"""
Portfolio Co-Pilot demo bot (Telegram, aiogram 3).
Educational demo only. Not investment advice.

Maps to the Pre-Sales Engineer job tasks:
  /needs      -> technical needs analysis + draft solution outline
  /profile    -> client risk profiling (demo flow)
  /portfolio  -> optimisation (mean-variance) + AI explanation
  /scenario   -> scenario / stress analysis
  /rebalance  -> drift check + rebalancing suggestion
  /faq        -> trial troubleshooting
  /feedback   -> client feedback collection (stored in SQLite)
"""
import asyncio
import json
import logging
import os
import sqlite3
import time
import warnings

import pandas as pd
import yfinance as yf
from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (BotCommand, CallbackQuery, InlineKeyboardButton,
                           InlineKeyboardMarkup, Message)
from dotenv import load_dotenv
from google import genai
from pypfopt import EfficientFrontier, expected_returns, risk_models

load_dotenv()
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore", message=".*inplace keyword.*")  # hide harmless yfinance/pandas noise
yf.set_tz_cache_location("yf_cache")  # private cache folder avoids 'database is locked'

BOT_TOKEN = os.environ["BOT_TOKEN"]
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
# Backup models tried in order if the main one is busy or fails (comma-separated)
GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash,gemini-3.5-flash-lite")
llm_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

DISCLAIMER = "Educational demo only. Not investment advice. Past performance does not predict future results."
UNIVERSE = ["SPY", "QQQ", "2800.HK", "AGG", "GLD", "VNQ"]
DRIFT_THRESHOLD = 0.05  # 5 percentage points

QUESTIONS = [
    ("1/4 What is your main goal?",
     [("Preserve capital", 1), ("Steady growth", 2), ("Maximise growth", 3)]),
    ("2/4 Investment horizon?",
     [("Under 3 years", 1), ("3 to 10 years", 2), ("10+ years", 3)]),
    ("3/4 If your portfolio fell 20%, you would...",
     [("Sell to stop losses", 1), ("Hold", 2), ("Buy more", 3)]),
    ("4/4 Investing experience?",
     [("Beginner", 1), ("Some", 2), ("Experienced", 3)]),
]

# Illustrative shock assumptions per asset (NOT forecasts)
SCENARIOS = {
    "crash": ("Equity market -20%",
              {"SPY": -.20, "QQQ": -.25, "2800.HK": -.22, "AGG": .02, "GLD": .05, "VNQ": -.18}),
    "rates": ("Interest rates +1%",
              {"SPY": -.06, "QQQ": -.10, "2800.HK": -.05, "AGG": -.05, "GLD": -.03, "VNQ": -.12}),
    "rally": ("Equity rally +15%",
              {"SPY": .15, "QQQ": .20, "2800.HK": .15, "AGG": -.01, "GLD": -.02, "VNQ": .12}),
}

FAQ = """Trial troubleshooting FAQ

1. Bot says "run /profile first"
   Portfolio, scenario and rebalance need a saved risk profile. Complete the 4 questions.

2. Portfolio takes a long time or fails
   Market data is fetched live. Wait ~20s and retry; if it persists, the data provider may be rate-limiting.

3. Explanation says "AI unavailable"
   The LLM service did not respond. Numbers are still valid; retry in a minute.

4. Numbers differ from my own tool
   The demo uses 3 years of history, a 6-ETF universe and a 3% risk-free rate. Different inputs give different results.

5. Want to change my profile?
   Run /profile again. Your old portfolio is replaced.

Anything else: use /feedback."""


# ---------- storage ----------
db = sqlite3.connect("copilot.db", check_same_thread=False)
db.executescript("""
CREATE TABLE IF NOT EXISTS users(
    user_id INTEGER PRIMARY KEY, score INTEGER, profile TEXT, weights TEXT);
CREATE TABLE IF NOT EXISTS feedback(
    id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, kind TEXT, text TEXT,
    ts DATETIME DEFAULT CURRENT_TIMESTAMP);
""")


def get_user(uid):
    row = db.execute("SELECT score, profile, weights FROM users WHERE user_id=?", (uid,)).fetchone()
    if not row:
        return None
    return {"score": row[0], "profile": row[1], "weights": json.loads(row[2]) if row[2] else None}


def save_profile(uid, score, profile):
    db.execute("INSERT INTO users(user_id, score, profile, weights) VALUES(?,?,?,NULL) "
               "ON CONFLICT(user_id) DO UPDATE SET score=?, profile=?, weights=NULL",
               (uid, score, profile, score, profile))
    db.commit()


def save_weights(uid, weights):
    db.execute("UPDATE users SET weights=? WHERE user_id=?", (json.dumps(weights), uid))
    db.commit()


def save_feedback(uid, kind, text):
    db.execute("INSERT INTO feedback(user_id, kind, text) VALUES(?,?,?)", (uid, kind, text))
    db.commit()


# ---------- finance logic (blocking; run in threads) ----------
def profile_from_score(score):
    if score <= 6:
        return "Conservative"
    return "Balanced" if score <= 9 else "Growth"


def download_prices(tickers, period):
    """Download closing prices safely: no threads (avoids 'database is locked'), retries, validation."""
    last_err = None
    for _ in range(3):
        try:
            data = yf.download(tickers, period=period, auto_adjust=True,
                               progress=False, threads=False)["Close"]
            if isinstance(data, pd.Series):
                data = data.to_frame(tickers[0])
            data = data.dropna(axis=1, how="all")
            missing = [t for t in tickers if t not in data.columns]
            if not missing:
                data = data.ffill().dropna()  # ffill handles different HK/US holidays
                if len(data) > 60:
                    return data
                last_err = f"only {len(data)} rows of data"
            else:
                last_err = f"no data for {missing}"
        except Exception as e:
            last_err = e
        time.sleep(2)
    raise RuntimeError(f"Could not download prices: {last_err}")


def build_portfolio(profile):
    prices = download_prices(UNIVERSE, "3y")
    mu = expected_returns.mean_historical_return(prices)
    cov = risk_models.sample_cov(prices)
    cap = {"Conservative": 0.35, "Balanced": 0.40, "Growth": 0.55}[profile]
    ef = EfficientFrontier(mu, cov, weight_bounds=(0.0, cap))
    if profile == "Conservative":
        ef.min_volatility()
    else:
        try:
            ef.max_sharpe(risk_free_rate=0.03)
        except ValueError:  # no asset beats the risk-free rate: fall back to lowest risk
            ef = EfficientFrontier(mu, cov, weight_bounds=(0.0, cap))
            ef.min_volatility()
    weights = {k: v for k, v in ef.clean_weights().items() if v > 0}
    ret, vol, sharpe = ef.portfolio_performance(risk_free_rate=0.03)
    return {"weights": weights, "ret": ret, "vol": vol, "sharpe": sharpe}


def compute_drift(weights):
    """Assume the client invested at target weights 6 months ago; see where weights drifted."""
    tickers = list(weights)
    px = download_prices(tickers, "6mo")
    growth = px.iloc[-1] / px.iloc[0]
    value = pd.Series(weights) * growth[tickers]
    current = value / value.sum()
    return {t: (weights[t], float(current[t])) for t in tickers}


async def llm(prompt):
    """Ask Gemini. Retries on temporary errors (503/429), then tries the optional fallback model."""
    if not llm_client:
        return "AI explanation unavailable (no GEMINI_API_KEY set)."

    models = [GEMINI_MODEL] + [m.strip() for m in GEMINI_FALLBACK_MODEL.split(",") if m.strip()]

    def call(model):
        return llm_client.models.generate_content(model=model, contents=prompt).text

    for model in models:
        for attempt in range(3):
            try:
                return await asyncio.to_thread(call, model)
            except Exception as e:
                logging.warning("Gemini %s attempt %d failed: %s", model, attempt + 1, e)
                if getattr(e, "code", None) in (429, 500, 503, 504):
                    if attempt < 2:
                        await asyncio.sleep(2 * (attempt + 1))  # wait 2s, then 4s, then retry
                    continue
                break  # permanent error (wrong model name, bad key): skip to next model
    return ("AI explanation unavailable right now (the AI service is busy). "
            "The numbers above are still valid. Try the command again in a minute.")


def fmt_weights(w):
    return "\n".join(f"{t}: {v:.0%}" for t, v in sorted(w.items(), key=lambda x: -x[1]))


# ---------- handlers ----------
router = Router()
answers: dict[int, list[int]] = {}


class Needs(StatesGroup):
    firm = State()
    pain = State()
    tools = State()


class Fb(StatesGroup):
    text = State()


def question_view(qi):
    text, opts = QUESTIONS[qi]
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=label, callback_data=f"q:{qi}:{score}")] for label, score in opts])
    return text, kb


@router.message(CommandStart())
async def start(m: Message, state: FSMContext):
    await state.clear()
    await m.answer(
        "PortfolioPal - AI portfolio co-pilot demo\n\n"
        "A demo of an AI co-pilot for wealth managers: profiling, optimisation, scenarios, "
        "rebalancing and plain-language explanations.\n\n"
        "/profile - risk questionnaire\n/portfolio - build portfolio\n/scenario - stress test\n"
        "/rebalance - drift check\n/needs - tell us about your firm\n/faq - troubleshooting\n"
        "/feedback - share feedback\n\n" + DISCLAIMER)


@router.message(Command("profile"))
async def profile_cmd(m: Message):
    answers[m.from_user.id] = []
    text, kb = question_view(0)
    await m.answer(text, reply_markup=kb)


@router.callback_query(F.data.startswith("q:"))
async def on_answer(cb: CallbackQuery):
    _, qi, s = cb.data.split(":")
    qi, s = int(qi), int(s)
    uid = cb.from_user.id
    ans = answers.setdefault(uid, [])
    if len(ans) != qi:  # stale or duplicate tap
        await cb.answer()
        return
    ans.append(s)
    if qi + 1 < len(QUESTIONS):
        text, kb = question_view(qi + 1)
        await cb.message.edit_text(text, reply_markup=kb)
    else:
        score = sum(ans)
        profile = profile_from_score(score)
        save_profile(uid, score, profile)
        answers.pop(uid, None)
        await cb.message.edit_text(
            f"Risk score: {score}/12\nProfile: {profile}\n\nNext: /portfolio")
    await cb.answer()


@router.message(Command("portfolio"))
async def portfolio_cmd(m: Message):
    user = get_user(m.from_user.id)
    if not user:
        await m.answer("Run /profile first.")
        return
    await m.answer("Optimising your portfolio (fetching market data)...")
    try:
        res = await asyncio.to_thread(build_portfolio, user["profile"])
    except Exception:
        logging.exception("optimisation failed")
        await m.answer("Could not fetch market data or optimise. See /faq and retry.")
        return
    save_weights(m.from_user.id, res["weights"])
    await m.answer(
        f"{user['profile']} portfolio\n\n{fmt_weights(res['weights'])}\n\n"
        f"Historical expected return: {res['ret']:.1%}\n"
        f"Volatility: {res['vol']:.1%}\nSharpe: {res['sharpe']:.2f}\n\n"
        "Based on 3 years of past data. " + DISCLAIMER)
    explanation = await llm(
        f"You are a wealth adviser's AI co-pilot. Explain this {user['profile']} portfolio to a client "
        f"in plain language, under 120 words, plain text, no markdown, no guarantees, no specific "
        f"buy/sell advice. Weights: {res['weights']}. Return {res['ret']:.1%}, volatility "
        f"{res['vol']:.1%}, Sharpe {res['sharpe']:.2f}. Tickers: SPY US large cap, QQQ US tech, "
        f"2800.HK Hong Kong equities, AGG US bonds, GLD gold, VNQ US real estate.")
    await m.answer("Co-pilot explanation:\n\n" + explanation)


@router.message(Command("scenario"))
async def scenario_cmd(m: Message):
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=name, callback_data=f"s:{key}")] for key, (name, _) in SCENARIOS.items()])
    await m.answer("Choose a scenario:", reply_markup=kb)


@router.callback_query(F.data.startswith("s:"))
async def on_scenario(cb: CallbackQuery):
    user = get_user(cb.from_user.id)
    if not user or not user["weights"]:
        await cb.message.answer("Build a portfolio first: /profile then /portfolio.")
        await cb.answer()
        return
    name, shocks = SCENARIOS[cb.data.split(":")[1]]
    lines, total = [], 0.0
    for t, w in user["weights"].items():
        contrib = w * shocks[t]
        total += contrib
        lines.append(f"{t}: weight {w:.0%}, shock {shocks[t]:+.0%}, impact {contrib:+.1%}")
    await cb.message.answer(
        f"Scenario: {name}\n\n" + "\n".join(lines) +
        f"\n\nEstimated portfolio impact: {total:+.1%}\n"
        "Shocks are illustrative assumptions, not forecasts.")
    await cb.answer()


@router.message(Command("rebalance"))
async def rebalance_cmd(m: Message):
    user = get_user(m.from_user.id)
    if not user or not user["weights"]:
        await m.answer("Build a portfolio first: /profile then /portfolio.")
        return
    await m.answer("Checking drift (assuming you invested at target weights 6 months ago)...")
    try:
        drift = await asyncio.to_thread(compute_drift, user["weights"])
    except Exception:
        logging.exception("drift failed")
        await m.answer("Could not fetch market data. See /faq and retry.")
        return
    lines, flagged = [], []
    for t, (target, current) in drift.items():
        diff = current - target
        mark = " <- over threshold" if abs(diff) > DRIFT_THRESHOLD else ""
        if mark:
            flagged.append(f"{t} {diff:+.1%}")
        lines.append(f"{t}: target {target:.0%}, now {current:.1%} ({diff:+.1%}){mark}")
    verdict = ("Rebalance suggested: " + ", ".join(flagged)) if flagged else \
              f"All positions within the {DRIFT_THRESHOLD:.0%} threshold. No rebalance needed."
    await m.answer("\n".join(lines) + "\n\n" + verdict)
    if flagged:
        text = await llm(
            "As an adviser's AI co-pilot, explain in under 100 words, plain text, why rebalancing "
            f"matters and what this drift means: {flagged}. No guarantees or specific advice.")
        await m.answer("Co-pilot explanation:\n\n" + text)


@router.message(Command("faq"))
async def faq_cmd(m: Message):
    await m.answer(FAQ)


# ----- pre-sales: needs analysis -----
@router.message(Command("needs"))
async def needs_cmd(m: Message, state: FSMContext):
    await state.set_state(Needs.firm)
    await m.answer("Needs analysis (3 questions).\n\n1/3 Tell me about your firm: size, number of clients, "
                   "and location.")


@router.message(Needs.firm)
async def needs_firm(m: Message, state: FSMContext):
    await state.update_data(firm=m.text)
    await state.set_state(Needs.pain)
    await m.answer("2/3 What is your biggest pain point in portfolio construction or client reporting today?")


@router.message(Needs.pain)
async def needs_pain(m: Message, state: FSMContext):
    await state.update_data(pain=m.text)
    await state.set_state(Needs.tools)
    await m.answer("3/3 Which tools or systems do you use now (CRM, portfolio system, spreadsheets)?")


@router.message(Needs.tools)
async def needs_tools(m: Message, state: FSMContext):
    data = await state.get_data()
    await state.clear()
    summary = f"Firm: {data['firm']} | Pain: {data['pain']} | Tools: {m.text}"
    save_feedback(m.from_user.id, "needs", summary)
    await m.answer("Thanks. Drafting a solution outline...")
    outline = await llm(
        "You are a pre-sales engineer for an AI wealth-technology company whose product is an AI "
        "co-pilot for advisers (personalised portfolios, predictive analytics, optimisation, dynamic "
        "rebalancing). Based on this client information, write a short solution outline in plain "
        "text, no markdown, under 200 words: (1) your understanding of their needs, (2) proposed "
        "modules, (3) a 2-week trial plan, (4) three follow-up discovery questions. "
        f"Client info: {summary}")
    await m.answer(outline)


# ----- feedback -----
@router.message(Command("feedback"))
async def feedback_cmd(m: Message, state: FSMContext):
    await state.set_state(Fb.text)
    await m.answer("What worked, what was confusing, what is missing? Send one message.")


@router.message(Fb.text)
async def feedback_save(m: Message, state: FSMContext):
    save_feedback(m.from_user.id, "feedback", m.text)
    await state.clear()
    await m.answer("Thank you, feedback recorded.")


async def main():
    bot = Bot(BOT_TOKEN)
    dp = Dispatcher()
    dp.include_router(router)
    await bot.set_my_commands([
        BotCommand(command="profile", description="Risk questionnaire"),
        BotCommand(command="portfolio", description="Build portfolio"),
        BotCommand(command="scenario", description="Stress test"),
        BotCommand(command="rebalance", description="Drift check"),
        BotCommand(command="needs", description="Needs analysis"),
        BotCommand(command="faq", description="Troubleshooting"),
        BotCommand(command="feedback", description="Send feedback"),
    ])
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())