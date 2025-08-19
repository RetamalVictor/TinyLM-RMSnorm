import requests, os
os.makedirs('data', exist_ok=True)
url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
text = requests.get(url).text
n = int(len(text)*0.9)
with open('data/tinyshakespeare_train.txt','w') as f: f.write(text[:n])
with open('data/tinyshakespeare_val.txt','w') as f: f.write(text[n:])
print('Wrote TinyShakespeare train/val.')