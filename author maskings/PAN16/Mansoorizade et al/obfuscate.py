import nltk
from nltk.corpus import brown
from nltk import data as nltk_data
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer as lemmatizer
import kenlm
import argparse
import os
import io
import codecs
import json
import random
from collections import Counter
from shutil import copyfile
import pattern.en
#dl = nltk.downloader.Downloader("http://nltk.github.com/nltk_data/")
#dl.download('punkt')
obfuse_id = 1
lm = None
def obfuscate(filePath, original_dist, changesPerSentence):
    global obfuse_id
    print("working on " + filePath)
    obfuse_id = 1
    p = codecs.open(filePath,"r", encoding='utf-8').read()
    parts = get_parts(p)
    for p in parts:
        for i in range(0,changesPerSentence):
            obfuscateSentence(p, original_dist)
    return parts
def get_parts(problem):
    global obfuse_id
    sentences = nltk.sent_tokenize(problem)
    curr = 0
    prev = 0
    text = u''
    parts = []
    for s in sentences:
        curr = problem.find(s,curr)
        assert curr > -1
        pad = problem[prev:curr]
        text = text + pad + s
        part = {  
            "original": pad + s,
            "original-start-charpos": prev,
            "original-end-charpos": curr + len(s) - 1,
            "obfuscation": s,
            "obfuscation-id": obfuse_id
        }
        parts.append(part)
        prev = curr + len(s)
        curr = curr + 1
        obfuse_id = obfuse_id + 1
    #assert problem == text
    return parts

def obfuscateSentence(part, original_dist):
    sentence = part['obfuscation']
    words = nltk.word_tokenize(sentence)
    sentence = ' '.join(words)
    new_sentence = u''
    best = {'word':words[0], 'score':-10000}
    bestIndex = 0
    i = 0
    tags = nltk.pos_tag(words)
    original_words = [w for w in original_dist.freqdist()]
    for t in tags:
        w = t[0]
        #if t[1] == "RB" and w[-2:] == "ly":
        #    w = words[i] = ""
        if w in original_words:# and random.randint(0,100) < 70:
            r = get_best_replacement(words, i, w, t[1]) 
            r['score'] *= original_dist.prob(w)
            if (best is None or r['score'] > best['score']) and w.lower() != r['word'].lower():
                best = r
                bestIndex = i
        i = i + 1

    words[bestIndex] = best['word']
    new_sentence = ' '.join(words)
    part["obfuscation"] = new_sentence

def adjust(w,POS):
   if POS == "NNS":
       return pattern.en.pluralize(pattern.en.singularize(w))
   if POS == "VB":
       return pattern.en.conjugate(w, 'inf')
   if POS == "VBP":
       return pattern.en.conjugate(w, '2sg') #'1sg'
   if POS == "VBZ":
       return pattern.en.conjugate(w, '3sg')
   if POS == "VBG":
       return pattern.en.conjugate(w, 'part')
   if POS == "VBD":
       return pattern.en.conjugate(w, '2sgp')
   if POS == "VBN":
       return pattern.en.conjugate(w, 'ppart')

   return None

def get_best_replacement(words, i, w, pos):
    syns = get_syns(w, pos)
    best = -0.05
    new_word = w
    for s in syns:
        a = adjust(s[0], pos)
        if a is None:
            a = s[0]
        diff = get_score_diff(words, i, a)
        if s[1] is None:
            diff = 0
        else:
            diff *= s[1]
        if diff > best:
            new_word = a
            best = diff
    if w == new_word:
        best = 0
    if best != 0 :
        best += 0.1
    return {'word':new_word, 'score': best}

def get_score_diff(words, i, new_word):
    global lm
    orig = ' '.join(words)
    w = words[i]
    words[i] = new_word
    new = ' '.join(words)
    words[i] = w

    return get_score(new) - get_score(orig)
def get_score(s):
    global lm
    return lm.score(s)

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return None
def get_syns(w, pos):
    p = get_wordnet_pos(pos)
    if p is None:
        return [[w,1.0]]
    synset = wn.synsets(w, p)
    syn = [[x[1],wn.wup_similarity(synset[0], x[1])]  for x in enumerate(synset)
           if not w.lower().startswith(x[1].name().split('.')[0])][:3]
    syn = sorted(syn, key=lambda x: x[1], reverse=True)
    return [[x[0].name().split('.')[0].replace("_", " "), x[1]] for x in syn]

def get_word_dist(text):
    words = nltk.word_tokenize(text)
    freq = nltk.FreqDist(words).most_common(200)
    dist = nltk.MLEProbDist(freq)
    return dist     
   
def save_json(obj, filePath):
    with codecs.open(filePath[0:len(filePath) - 4] + ".json","w",'utf-8') as f:
        json.dump(obj, f, indent=4, sort_keys=True)

def save_text(obj, filePath):
    with codecs.open(filePath,"wa",'utf-8') as f:
        for p in obj:
            f.write(p["obfuscation"] + " ")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help="path to input dataset")
    parser.add_argument('-o', '--output', help="path to output directory")
    parser.add_argument('-lm', help="language model file", default="corpus.binary")
    parser.add_argument('-t', help="save obfuscated texts", default="false")
    parser.add_argument('-j', help="save obfuscated JSON", default="true")
    args = parser.parse_args()
    if args.input is None or args.output is None:
        parser.print_usage()
        exit()
    return args

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def main():
    args = get_args()
    output = os.path.normpath(args.output)
    input = os.path.normpath(args.input)
    save_texts = (args.t != "false")
    save_jsons = (args.j == "true")
    global lm
    lm = kenlm.LanguageModel(args.lm)
    mkdir(output)
    count = 0
    for path in os.listdir(input):
        in_dir = os.path.join(input, path)
        if os.path.isdir(in_dir):
            out_dir = os.path.join(output, path)
            mkdir(out_dir)
            problems = sorted(os.listdir(in_dir))
            original_dist = []
            auther_texts = ""
            for problem in problems:
                problem_path = os.path.join(in_dir, problem)
                if os.path.isfile(problem_path):
                    if problem != 'original.txt':
                        with codecs.open(problem_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                            auther_texts = auther_texts + " " + text
            print("------------------------------------")
            original_dist = get_word_dist(auther_texts)
            #if save_texts:
            #    copyfile(problem_path, output_path)
            problem_path = os.path.join(in_dir, "original.txt")
            output_path = os.path.join(out_dir, "obfuscation.txt")
            obj = obfuscate(problem_path, original_dist,1)
            if save_jsons:
                print("saving JSON ....")
                save_json(obj, output_path)
            if save_texts:
                save_text(obj, output_path)

main()
