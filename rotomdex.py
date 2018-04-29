import pickle
import re
import operator
import discord
import yaml
import numpy
import keras
import pythainlp

def get_index(word, word2idx):
    return word2idx[word] if word in word2idx else word2idx['UNK']

def clean(message):
    tags = re.compile(r'<[^>]+>')
    spaces = re.compile(r'\s{2,}')
    return spaces.sub(' ', tags.sub('', message.strip()).strip())

def preprocess(message, word2idx, maxlen=1000):
    cleaned = clean(message)
    tokenized = pythainlp.tokenize.word_tokenize(cleaned, engine='deepcut')
    input_x = numpy.array([[get_index(word, word2idx) for word in tokenized]])
    input_x = keras.preprocessing.sequence.pad_sequences(input_x, maxlen=maxlen, dtype='int32', padding='post', truncating='pre', value=0.)
    return input_x

def class2name(cls):
    names = ['Bulbasaur', 'Charmander', 'Squirtle', 'Pikachu', 'Meowth', 'Jigglypuff', 'Zubat', 'Onix', 'Magikarp', 'Chikorita']
    return names[cls] if cls < len(names) else 'Unknown'

def postprocess(predictions):
    probs = predictions[0]
    classes = numpy.argsort(probs)[::-1]
    results = [(class2name(cls), probs[cls]) for cls in classes]
    return results

def format(results, metadata):
    name = results[0][0]
    if metadata:
        title = metadata[name]['title']
        description = metadata[name]['description']
        url = metadata[name]['url']
        image = metadata[name]['image']
        color = metadata[name]['color']
    else:
        title = name
        description = discord.Embed.Empty
        url = 'https://bulbapedia.bulbagarden.net/wiki/' + name
        image = discord.Embed.Empty
        color = discord.Embed.Empty
    footer = ', '.join(['{} ({:0.4f})'.format(result[0], result[1]) for result in results])
    return discord.Embed(title=title, description=description, url=url).set_image(url=image).set_footer(text=footer)

print('loading config')
with open('config.yml', 'r') as f:
    config = yaml.load(f)

print('loading word2idx')
with open(config['word_to_index'], 'rb') as f:
    word2idx = pickle.load(f)

print('loading model')
pokemon_model = keras.models.load_model(config['model'])

print('loading metadata')
with open(config['metadata'], 'rb') as f:
    pokemon_metadata = pickle.load(f)

print('connecting to Discord')
client = discord.Client()

@client.event
async def on_message(message):
    if message.author == client.user:
        return
    if client.user.mentioned_in(message):
        await client.send_typing(message.channel)
        input_x = preprocess(message.content, word2idx)
        predictions = pokemon_model.predict(input_x)
        results = postprocess(predictions)
        await client.send_message(message.channel, embed=format(results[:3], pokemon_metadata))

@client.event
async def on_ready():
    print('Logged in as')
    print(client.user.name)
    print(client.user.id)
    print('------')

client.run(config['token'])
