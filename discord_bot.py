import discord
from discord.ext import commands
import requests
import json
import yaml

# load discord token from config.yaml
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)
    DISCORD_TOKEN = config["DISCORD_TOKEN"]
    ROASTING_API_URL = config["ROASTING_API_URL"]

# Create a new client instance
intents = discord.Intents.default()
intents.message_content = True  # Enable reading message content

bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user.name} - {bot.user.id}")
    print("------")

async def roast_code(code_snippet: str) -> str:
    payload = {
        "code": code_snippet
    }

    # Send the POST request
    response = requests.post(ROASTING_API_URL, json=payload)
    json_data = json.loads(response.text)

    roasting_response = json_data["generated_text"][1]["content"]

    # Print the response
    return roasting_response

def chunk_message(message: str, chunk_size=2000):
    return [message[i:i+chunk_size] for i in range(0, len(message), chunk_size)]

@bot.command(name='roast')
async def roast(ctx):
    # Ensure the command is a reply
    if not ctx.message.reference:
        await ctx.send("You need to reply to the code you want roasted.")
        return
    
    # Get the replied message
    replied_message = await ctx.channel.fetch_message(ctx.message.reference.message_id)
    code_text = replied_message.content

    # Initial placeholder response
    message = await ctx.reply("🧠 Roasting in progress... this might take a minute.")

    # Send to your code roasting API
    roast_response = await roast_code(code_text)
    print(roast_response)

    chunks = chunk_message(roast_response)

    for chunk in chunks:
        await ctx.send(chunk)

bot.run(DISCORD_TOKEN)