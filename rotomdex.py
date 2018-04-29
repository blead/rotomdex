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
    regex = re.compile(r'<[^>]+>')
    return regex.sub('', message.strip())

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
    result = numpy.apply_along_axis(lambda x: max(enumerate(x), key=operator.itemgetter(1)), 1, predictions)[0]
    print(result)
    result = (class2name(int(result[0])), result[1])
    print(result)
    return result

print('loading config')
with open('config.yml', 'r') as f:
    config = yaml.load(f)

print('loading word2idx')
with open(config['word_to_index'], 'rb') as f:
    word2idx = pickle.load(f)

print('loading model')
pokemon_model = keras.models.load_model(config['model'])

print('connecting to Discord')
client = discord.Client()

@client.event
async def on_message(message):
    if message.author == client.user:
        return
    if client.user.mentioned_in(message):
        input_x = preprocess(message.content, word2idx)
        predictions = pokemon_model.predict(input_x)
        result = postprocess(predictions)
        await client.send_message(message.channel, '{} ({} confidence)'.format(result[0], result[1]))
        # await client.send_message(message.channel, 'https://bulbapedia.bulbagarden.net/wiki/' + query)

@client.event
async def on_ready():
    print('Logged in as')
    print(client.user.name)
    print(client.user.id)
    print('------')

client.run(config['token'])
