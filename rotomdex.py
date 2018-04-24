import discord
import yaml

client = discord.Client()

@client.event
async def on_message(message):
    # ignore own message
    if message.author == client.user:
        return
    if message.content == 'ping':
        await client.send_message(message.channel, 'pong')

@client.event
async def on_ready():
    print('Logged in as')
    print(client.user.name)
    print(client.user.id)
    print('------')


with open('config.yml', 'r') as f:
    config = yaml.load(f)

client.run(config['token'])
