import nltk
import io
from nltk.corpus import brown
def main():
    f = open('corpus.txt', 'w');
    text = ' '.join(brown.words())
    f.write(text)
    f.close()
main()
