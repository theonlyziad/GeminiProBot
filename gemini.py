import os
import io
import time
import logging
import PIL.Image
from pyrogram.types import Message
from pyrogram import Client, filters
from pyrogram.enums import ParseMode
import google.generativeai as genai

from config import API_ID, API_HASH, BOT_TOKEN, GOOGLE_API_KEY, MODEL_NAME

app = Client(
    "gemini_session",
    api_id=API_ID,
    api_hash=API_HASH,
    bot_token=BOT_TOKEN,
    parse_mode=ParseMode.MARKDOWN
)

genai.configure(api_key=GOOGLE_API_KEY)
client = genai.Client()

@app.on_message(filters.command("gem"))
async def gemi_handler(client_app, message: Message):
    loading = await message.reply_text("**Generating response, please wait...**")
    try:
        if len(message.text.strip()) <= 5:
            await message.reply_text("**Provide a prompt after the command.**")
            return
        prompt = message.text.split(maxsplit=1)[1]
        # Generate text as before
        response = client.models.generate_text(model=MODEL_NAME, prompt=prompt)
        await message.reply_text(response.text)
    except Exception as e:
        await message.reply_text(f"**Error: {e}**")
    finally:
        await loading.delete()

@app.on_message(filters.command("veo"))
async def veo_handler(client_app, message: Message):
    loading = await message.reply_text("**Generating video, please wait...**")
    try:
        if len(message.text.strip()) <= 4:
            await message.reply_text("**Provide a prompt after the command.**")
            return
        prompt = message.text.split(maxsplit=1)[1]
        operation = client.models.generate_videos(
            model="veo-3.0-generate-preview",
            prompt=prompt
        )
        while not operation.done:
            await asyncio.sleep(5)
            operation = client.operations.get(operation)
        video_url = operation.result.video_url  # مثال افتراضي
        await message.reply_video(video_url)
    except Exception as e:
        await message.reply_text(f"**Error: {e}**")
    finally:
        await loading.delete()

@app.on_message(filters.command("imgveo"))
async def imgveo_handler(client_app, message: Message):
    loading = await message.reply_text("**Generating image-to-video, please wait...**")
    try:
        if not message.reply_to_message or not message.reply_to_message.photo:
            await message.reply_text("**Reply to a photo and include prompt.**")
            return
        prompt = message.command[1] if len(message.command) > 1 else message.reply_to_message.caption or ""
        img_data = await client_app.download_media(message.reply_to_message, in_memory=True)
        img = PIL.Image.open(io.BytesIO(img_data.getbuffer()))
        operation = client.models.generate_videos(
            model="veo-3.0-generate-preview",
            prompt=prompt,
            image=img
        )
        while not operation.done:
            await asyncio.sleep(5)
            operation = client.operations.get(operation)
        video_url = operation.result.video_url
        await message.reply_video(video_url)
    except Exception as e:
        logging.error(f"Error: {e}")
        await message.reply_text("**An error occurred. Please try again.**")
    finally:
        await loading.delete()

if __name__ == '__main__':
    app.run()
