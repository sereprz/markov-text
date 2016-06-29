import re
import nltk
import urllib

url = 'http://www.gutenberg.org/files/174/174.txt'
response = urllib.urlopen(url)
raw = response.read().decode('utf8')
start = raw.find('CHAPTER 1')
end = raw.rfind('End of Project Gutenberg\'s')
raw = raw[start:end]
raw = re.sub('CHAPTER [0-9]+', '', raw)
raw = re.sub('\r\n', ' ', raw)
