"""
Author: Ziad (@zeedtek)
All rights reserved.
"""

import os
import io
import re
import json
import time
import random
import string
import logging
import tempfile
from datetime import datetime, timedelta
from urllib.parse import quote_plus

import requests
import PIL.Image

from pyrogram import Client, filters
from pyrogram.enums import ParseMode
from pyrogram.types import (
    Message,
    InlineKeyboardMarkup,
    InlineKeyboardButton
)

import google.generativeai as genai

# ------------------ إعدادات عامة ------------------
from config import API_ID, API_HASH, BOT_TOKEN, GOOGLE_API_KEY, MODEL_NAME

DATA_FILE = "users.json"      # قاعدة البيانات
API_BASE_TXT2VIDEO = "https://api.yabes-desu.workers.dev/ai/tool/txt2video"
DEV_USERNAME_CONTACT = "theonlyziad"  # لزر الشراء بعد نهاية المجاني
DEV_MENTION = "@theonlyziad"
BRAND_PREFIX = "zeedtek"      # بادئة الأكواد
FREE_DAYS_DEFAULT = 3         # عدد أيام الفترة المجانية بدءاً من أول تشغيل

# المالك الوحيد (ID ثابت كما طلبت)
OWNER_ID = 5000510953

# ------------------ لوجينغ ------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("bot")

# ------------------ تهيئة Gemini ------------------
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel(MODEL_NAME)

# ------------------ أدوات تخزين ------------------
def _init_db():
    if not os.path.exists(DATA_FILE):
        # يحدد free_start وقت أول تشغيل ويخزن
        data = {
            "premium": {},        # { user_id: expiry_ts }
            "codes": {},          # { code: {grant_seconds, expires_at, redeemed} }
            "free_start": int(time.time())  # بداية الفترة المجانية من أول تشغيل
        }
        _save_db(data)
        return data
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_db(data: dict):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

db = _init_db()

def get_free_window():
    """إرجاع نافذة المجاني (start, end) من قاعدة البيانات"""
    start_ts = db.get("free_start", int(time.time()))
    start_dt = datetime.utcfromtimestamp(start_ts)
    end_dt = start_dt + timedelta(days=FREE_DAYS_DEFAULT)
    return start_dt, end_dt

# ------------------ أدوات وقت وصيغ ------------------
def humanize_seconds(seconds: int) -> str:
    seconds = max(0, int(seconds))
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, _ = divmod(rem, 60)
    parts = []
    if d: parts.append(f"{d} يوم")
    if h: parts.append(f"{h} ساعة")
    if m: parts.append(f"{m} دقيقة")
    return " و ".join(parts) if parts else "أقل من دقيقة"

DUR_PATTERN = re.compile(r"^\s*(\d+)\s*([A-Za-z]+)\s*$")

def parse_duration(token: str) -> int:
    """
    يحول 1Mn / 1H / 1d إلى ثواني.
    يدعم الأحرف الكبيرة/الصغيرة (Mn, H, d).
    """
    m = DUR_PATTERN.match(token or "")
    if not m:
        raise ValueError("صيغة المدة غير صحيحة. استعمل 1Mn / 1H / 1d")
    amount = int(m.group(1))
    unit = m.group(2).lower()
    if unit == "mn":
        return amount * 60
    if unit == "h":
        return amount * 3600
    if unit == "d":
        return amount * 86400
    raise ValueError("صيغة المدة غير صحيحة. استعمل 1Mn / 1H / 1d")

def is_admin(msg: Message) -> bool:
    # التحقق بالـ ID (أكثر أمانًا)
    return bool(msg.from_user and msg.from_user.id == OWNER_ID)

def now_utc() -> datetime:
    return datetime.utcnow()

def user_is_premium(user_id: int) -> bool:
    exp_ts = db["premium"].get(str(user_id))
    return bool(exp_ts and exp_ts > time.time())

def premium_expiry_dt(user_id: int) -> datetime | None:
    exp_ts = db["premium"].get(str(user_id))
    if not exp_ts:
        return None
    return datetime.utcfromtimestamp(exp_ts)

# ------------------ شبكة: فيديو من النص ------------------
def fetch_video_to_temp(prompt: str) -> str:
    url = f"{API_BASE_TXT2VIDEO}?prompt={quote_plus(prompt)}"
    resp = requests.get(url, stream=True, timeout=600)
    if resp.status_code != 200:
        # لو API يرجّع JSON يحتوي رابط فيديو
        ctype = resp.headers.get("Content-Type", "")
        if "application/json" in ctype:
            try:
                data = resp.json()
                video_url = data.get("url") or data.get("video") or data.get("result") or data.get("data")
                if not video_url:
                    raise RuntimeError(f"API error {resp.status_code}: no video url")
                r2 = requests.get(video_url, stream=True, timeout=600)
                r2.raise_for_status()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tf:
                    for chunk in r2.iter_content(1024 * 64):
                        tf.write(chunk)
                    return tf.name
            except Exception as e:
                raise RuntimeError(f"API JSON error: {e}")
        raise RuntimeError(f"API error {resp.status_code}: {resp.text[:200]}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tf:
        for chunk in resp.iter_content(1024 * 64):
            tf.write(chunk)
        return tf.name

# ------------------ بوت Pyrogram ------------------
app = Client(
    "gemini_session",
    api_id=API_ID,
    api_hash=API_HASH,
    bot_token=BOT_TOKEN,
    parse_mode=ParseMode.MARKDOWN
)

# ------------------ /start ------------------
@app.on_message(filters.command("start"))
async def start_handler(_, m: Message):
    free_start, free_end = get_free_window()
    free_text = (
        f"🎁 الفترة المجانية لصنع الفيديوهات مستمرة حتى: **{free_end.strftime('%Y-%m-%d %H:%M:%S')} UTC**"
        if now_utc() <= free_end else
        "⛔️ الفترة المجانية لصنع الفيديوهات انتهت."
    )
    msg = (
        "👋 أهلاً بك!\n\n"
        "🤖 اكتب أي نص وسأرد عليك مباشرةً بواسطة **Gemini**.\n"
        "📸 **/ai** — رد على صورة لتحليلها.\n"
        "🎬 **/veo <prompt>** — تحويل النص إلى فيديو.\n\n"
        f"{free_text}\n\n"
        "👨‍💻 Dev: @zeedtek"
    )
    await m.reply_text(msg)

# ------------------ /help ------------------
@app.on_message(filters.command("help"))
async def help_handler(_, m: Message):
    msg = (
        "🆘 **شرح البوت**\n\n"
        "• اكتب أي نص — يرد عليك Gemini مباشرة ✍️\n"
        "• `/ai` — رد على صورة ليحللها 📸\n"
        "• `/veo <وصف>` — إنشاء فيديو 🎬\n"
        "• `/redeem <code>` — تفعيل كود 🎟️ (يبدأ بـ zeedtek)\n"
        "• `/myplan` أو `/status` — معرفة حالتك والمدة المتبقية ⏳\n\n"
        "🔧 **أوامر الأدمن (ID فقط 5000510953):**\n"
        "• `/add <user_id> <مدة>` — إضافة بريميوم (1Mn / 1H / 1d)\n"
        "• `/gen <مدة_البريميوم> [مدة_صلاحية_الكود]` — توليد كود بريميوم (صلاحية الكود افتراضيًا 7d)\n"
        "• `/free_status` — حالة الفترة المجانية\n\n"
        "👨‍💻 Dev: @zeedtek"
    )
    await m.reply_text(msg)

# ------------------ الرد على النصوص (Gemini) ------------------
@app.on_message(filters.text & ~filters.command(["ai", "veo", "redeem", "gen", "add", "myplan", "status", "free_status", "start", "help"]))
async def gemini_text_handler(_, m: Message):
    prompt = (m.text or "").strip()
    if not prompt:
        return
    try:
        resp = gemini_model.generate_content(prompt)
        text = resp.text or " "
        # تقسيم لو طويل
        for i in range(0, len(text), 4000):
            await m.reply_text(text[i:i+4000])
    except Exception as e:
        log.exception("Gemini error")
        await m.reply_text(f"⚠️ حدث خطأ أثناء التوليد: `{e}`")

# ------------------ /ai تحليل الصور ------------------
@app.on_message(filters.command("ai"))
async def ai_image_handler(_, m: Message):
    if not m.reply_to_message or not (m.reply_to_message.photo or m.reply_to_message.document):
        await m.reply_text("**📸 استخدم `/ai` بالرد على صورة (مع وصف اختياري في نفس الأمر).**")
        return
    if len(m.command) > 1:
        prompt = " ".join(m.command[1:])
    elif m.reply_to_message.caption:
        prompt = m.reply_to_message.caption
    else:
        prompt = "Describe this image."

    try:
        file = await app.download_media(m.reply_to_message, in_memory=True)
        img = PIL.Image.open(io.BytesIO(file.getbuffer()))
        resp = gemini_model.generate_content([prompt, img])
        await m.reply_text(resp.text or " ")
    except Exception as e:
        log.exception("AI image error")
        await m.reply_text("⚠️ حدث خطأ أثناء تحليل الصورة. حاول مرة أخرى.")

# ------------------ /veo فيديو من وصف ------------------
@app.on_message(filters.command("veo"))
async def veo_handler(_, m: Message):
    if len(m.command) < 2:
        await m.reply_text("📝 اكتب وصف الفيديو بعد الأمر: `/veo a boy running in the rain cinematic 4k`", quote=True)
        return

    prompt = " ".join(m.command[1:])
    free_start, free_end = get_free_window()

    # يُسمح إن كان داخل الفترة المجانية أو عنده بريميوم
    if now_utc() <= free_end or user_is_premium(m.from_user.id):
        loading = await m.reply_text("🎬 **جاري إنشاء الفيديو... انتظر قليلاً**")
        try:
            video_path = fetch_video_to_temp(prompt)
            await m.reply_video(
                video=open(video_path, "rb"),
                caption=f"النص: {prompt}\n\n👨‍💻 Dev: @zeedtek",
                supports_streaming=True
            )
        except Exception as e:
            log.exception("VEO error")
            await m.reply_text(f"⚠️ خطأ أثناء إنشاء الفيديو:\n`{e}`")
        finally:
            try: await loading.delete()
            except: pass
    else:
        btn = InlineKeyboardMarkup([
            [InlineKeyboardButton("💎 تواصل للشراء", url=f"https://t.me/{DEV_USERNAME_CONTACT}")]
        ])
        await m.reply_text(
            "🚫 انتهت الفترة المجانية لصنع الفيديوهات.\n"
            f"تواصل مع {DEV_MENTION} للاشتراك في Premium.",
            reply_markup=btn
        )

# ------------------ /myplan و /status ------------------
@app.on_message(filters.command(["myplan", "status"]))
async def myplan_handler(_, m: Message):
    exp_dt = premium_expiry_dt(m.from_user.id)
    free_start, free_end = get_free_window()

    lines = []
    # حالة الفترة المجانية العامة
    if now_utc() <= free_end:
        remaining = (free_end - now_utc()).total_seconds()
        lines.append(f"🎁 الفترة المجانية سارية حتى: **{free_end.strftime('%Y-%m-%d %H:%M:%S')} UTC**")
        lines.append(f"⏳ المتبقي للمجاني: **{humanize_seconds(remaining)}**")
    else:
        lines.append("🎁 الفترة المجانية: **انتهت**")

    # حالة البريميوم للمستخدم
    if exp_dt and exp_dt > now_utc():
        remaining = (exp_dt - now_utc()).total_seconds()
        lines.append(f"\n💎 Premium فعال حتى: **{exp_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC**")
        lines.append(f"⏳ المتبقي لاشتراكك: **{humanize_seconds(remaining)}**")
    else:
        lines.append("\n💤 لا تملك اشتراك Premium حالياً.")

    await m.reply_text("\n".join(lines))

# ------------------ /free_status (أدمن) ------------------
@app.on_message(filters.command("free_status"))
async def free_status_handler(_, m: Message):
    if not is_admin(m):
        return
    free_start, free_end = get_free_window()
    now = now_utc()
    if now <= free_end:
        remaining = (free_end - now).total_seconds()
        text = (
            f"✅ الفترة المجانية **سارية**\n"
            f"• البداية: {free_start.strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
            f"• النهاية: {free_end.strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
            f"• المتبقي: {humanize_seconds(remaining)}"
        )
    else:
        text = (
            f"⛔ الفترة المجانية **انتهت**\n"
            f"• البداية: {free_start.strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
            f"• النهاية: {free_end.strftime('%Y-%m-%d %H:%M:%S')} UTC"
        )
    await m.reply_text(text)

# ------------------ /add (أدمن) ------------------
@app.on_message(filters.command("add"))
async def add_handler(_, m: Message):
    if not is_admin(m):
        return
    try:
        # /add <user_id> <duration>
        _, uid, dur = m.text.strip().split(maxsplit=2)
        seconds = parse_duration(dur)
        exp_old_ts = db["premium"].get(uid, 0)
        base_ts = exp_old_ts if exp_old_ts and exp_old_ts > time.time() else time.time()
        new_exp_ts = int(base_ts + seconds)
        db["premium"][uid] = new_exp_ts
        _save_db(db)

        remaining = new_exp_ts - int(time.time())
        exp_dt = datetime.utcfromtimestamp(new_exp_ts)
        await m.reply_text(
            f"✅ Premium مُضاف للمستخدم `{uid}`\n"
            f"⏳ المدة المتبقية: **{humanize_seconds(remaining)}**\n"
            f"📅 ينتهي في: **{exp_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC**",
            quote=True
        )
    except ValueError as ve:
        await m.reply_text(f"❌ {ve}\nمثال: `/add 123456789 1d`", quote=True)
    except Exception as e:
        log.exception("/add error")
        await m.reply_text(f"⚠️ خطأ: `{e}`", quote=True)

# ------------------ /gen (أدمن) ------------------
@app.on_message(filters.command("gen"))
async def gen_handler(_, m: Message):
    if not is_admin(m):
        return
    try:
        # صيغ مدعومة:
        # /gen <مدة_البريميوم>
        # /gen <مدة_البريميوم> <مدة_صلاحية_الكود>
        parts = m.text.strip().split()
        if len(parts) < 2:
            raise ValueError("الصيغة: /gen <مدة_البريميوم> [مدة_صلاحية_الكود]")

        grant_token = parts[1]              # مدة البريميوم للمستفيد (مثال: 1H)
        grant_seconds = parse_duration(grant_token)

        # صلاحية الكود نفسه (افتراضي 7 أيام)
        validity_seconds = parse_duration(parts[2]) if len(parts) >= 3 else 7 * 86400

        code = BRAND_PREFIX + ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        db["codes"][code] = {
            "grant_seconds": int(grant_seconds),
            "expires_at": int(time.time()) + int(validity_seconds),
            "redeemed": False
        }
        _save_db(db)

        exp_human = humanize_seconds(validity_seconds)
        grant_human = humanize_seconds(grant_seconds)
        await m.reply_text(
            "🎟️ كود جديد:\n"
            f"`{code}`\n"
            f"🎁 يمنح: **{grant_human}** بريميوم\n"
            f"⏳ صالح لمدة: **{exp_human}** من الآن",
            quote=True
        )
    except ValueError as ve:
        await m.reply_text(f"❌ {ve}\nأمثلة:\n`/gen 1H`\n`/gen 1d 2d`", quote=True)
    except Exception as e:
        log.exception("/gen error")
        await m.reply_text(f"⚠️ خطأ: `{e}`", quote=True)

# ------------------ /redeem ------------------
@app.on_message(filters.command("redeem"))
async def redeem_handler(_, m: Message):
    try:
        # /redeem <code>
        _, code = m.text.strip().split(maxsplit=1)
        code = code.strip()
        if not code.startswith(BRAND_PREFIX):
            await m.reply_text("❌ الكود غير صالح. يجب أن يبدأ بـ `zeedtek`.", quote=True)
            return

        info = db["codes"].get(code)
        if not info:
            await m.reply_text("❌ الكود غير موجود أو تم استخدامه مسبقاً.", quote=True)
            return

        if info.get("redeemed"):
            await m.reply_text("⚠️ هذا الكود تم استخدامه مسبقاً.", quote=True)
            return

        if time.time() > info.get("expires_at", 0):
            await m.reply_text("⏰ انتهت صلاحية هذا الكود.", quote=True)
            # يمكنك حذف الكود المنتهي تلقائياً إن رغبت
            return

        grant_seconds = int(info.get("grant_seconds", 0))
        if grant_seconds <= 0:
            await m.reply_text("⚠️ هذا الكود لا يحتوي مدة صالحة.", quote=True)
            return

        uid = str(m.from_user.id)
        exp_old_ts = db["premium"].get(uid, 0)
        base_ts = exp_old_ts if exp_old_ts and exp_old_ts > time.time() else time.time()
        new_exp_ts = int(base_ts + grant_seconds)
        db["premium"][uid] = new_exp_ts

        # علّم الكود كمستخدم
        db["codes"][code]["redeemed"] = True
        _save_db(db)

        remaining = new_exp_ts - int(time.time())
        exp_dt = datetime.utcfromtimestamp(new_exp_ts)
        await m.reply_text(
            "✅ تم تفعيل Premium لك!\n"
            f"🎁 المدة المضافة: **{humanize_seconds(grant_seconds)}**\n"
            f"⏳ المتبقي الآن: **{humanize_seconds(remaining)}**\n"
            f"📅 ينتهي في: **{exp_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC**",
            quote=True
        )
    except ValueError:
        await m.reply_text("❌ الاستعمال: `/redeem zeedtekXXXXXX`", quote=True)
    except Exception as e:
        log.exception("/redeem error")
        await m.reply_text(f"⚠️ خطأ: `{e}`", quote=True)

# ------------------ تشغيل البوت ------------------
if __name__ == "__main__":
    app.run()
