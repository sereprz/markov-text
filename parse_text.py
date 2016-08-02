import re
import urllib

START_STRING = 'CHAPTER 1'
END_STRING = 'End of Project Gutenberg\'s'
PART = 'CHAPTER'
DORIAN_GRAY = 'http://www.gutenberg.org/files/174/174.txt'


def parse_from_url(url,
                   start_str=START_STRING,
                   end_str=END_STRING,
                   unit=PART,
                   end_sentence='$'):
    '''
        Parse text from url (defaults from Project Gutenberg's)

        :url: str, url for the text to parse
        :start_str: str, Starting sentence for the text
        :end_str: str, End sentence for the text
        :unit: str, single unit to divide the text (e.g. chapters)

    '''

    response = urllib.urlopen(url)
    raw = response.read().decode('utf8')
    start = raw.find(start_str)
    end = raw.rfind(end_str)
    raw = raw[start:end]
    raw = re.sub(unit + ' [0-9]+', '', raw)
    raw = re.sub('\r\n', ' ', raw)

    sentences = []
    for line in re.split(re.compile('\.|!|\?|"'), raw):
        if line.strip() != '':
            sentences.append(line.strip())

    return end_sentence.join([s[:1].lower() + s[1:] for s in sentences])
