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

pokemon_names = ['Bulbasaur', 'Charmander', 'Squirtle', 'Pikachu', 'Meowth', 'Jigglypuff', 'Zubat', 'Onix', 'Magikarp', 'Chikorita']
pokemon_class = {name: i for i, name in enumerate(pokemon_names)}

def class2name(cls):
    return pokemon_names[cls] if cls < len(pokemon_names) else 'Unknown'

def name2class(name):
    name = name.capitalize()
    return pokemon_class[name] if name in pokemon_class else -1

def postprocess(predictions):
    probs = predictions[0]
    classes = numpy.argsort(probs)[::-1]
    results = [(class2name(cls), probs[cls]) for cls in classes]
    return results

def format(results):
    return '\n'.join(['{} ({} confidence)'.format(result[0], result[1]) for result in results])

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

last_query = ''

@client.event
async def on_message(message):
    global last_query
    if message.author == client.user:
        return
    if message.content.lower().startswith('!correct'):
        split_content = message.content.split(' ')
        if len(split_content) > 1:
            correct_name = split_content[1]
            correct_id = name2class(correct_name)
            with open('correct_log.txt', 'a+', encoding='utf8') as f:
                f.write('{}\t{}\n'.format(last_query, correct_id if correct_id > 0 else correct_name))
            await client.send_message(message.channel, 'ขอบคุณนะ เราจะจำไว้ ^^')
    elif client.user.mentioned_in(message):
        last_query = clean(message.content).strip()
        input_x = preprocess(message.content, word2idx)
        predictions = pokemon_model.predict(input_x)
        results = postprocess(predictions)
        await client.send_message(message.channel, format(results[:3]))
        # await client.send_message(message.channel, 'https://bulbapedia.bulbagarden.net/wiki/' + query)

@client.event
async def on_ready():
    print('Logged in as')
    print(client.user.name)
    print(client.user.id)
    print('------')

client.run(config['token'])
