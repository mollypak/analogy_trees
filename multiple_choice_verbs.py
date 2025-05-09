import json
with open('storyanalogy_multiple_choice.json') as f:
    multiple = json.load(f)
import stanza
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
nlp = stanza.Pipeline('en', processors='tokenize,pos')
from transformers import AutoTokenizer, BertModel

import gensim.downloader as api

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = BertModel.from_pretrained("google-bert/bert-base-uncased")
correct = 0
hard = 0
random = 0
for mult in multiple:
    answers = mult['choices']
    source = mult['source']
    source = source.replace("’s", " is")
    source = source.replace("'t", " not")

    v1 = []
    doc = nlp(source)
    for sent in doc.sentences:
        for item in sent.words:
            if (item.upos == 'VERB')|(item.upos == 'AUX')|(item.upos == 'VB'):
                v1.append(item.text.lower())
    ave_score = []
    for answer in answers:
        answer = answer.replace("’s", " is")
        answer = answer.replace("'t", " not")

        v2 = []
        doc = nlp(answer)
        for sent in doc.sentences:
            for item in sent.words:
                if (item.upos == 'VERB') | (item.upos == 'AUX') | (item.upos == 'VB'):
                    v2.append(item.text.lower())
        doc = nlp(answer)
        sub_prop = []
        v1 = list(set(v1))
        v2 = list(set(v2))
        sim = []
        words = []
        if v1 and v2:
            for verb in v1:
                if verb in v2:
                    sim.append(1)
                    words.append([verb, verb])
                    v2.remove(verb)
            if words:
                for w in words:
                    v1.remove(w[0])
        else:
            sim.append(-100)
            words.append(['empty'])
        if v1 and v2:
            embed1 = []
            embed2 = []
            for verb in v1:
                embed = model(tokenizer(verb, return_tensors="pt")['input_ids']).last_hidden_state[:, 1:-1, :].squeeze()
                if embed.dim() > 1:
                    embed = sum(embed) / len(embed)
                embed1.append(embed)
            for verb in v2:
                embed = model(tokenizer(verb, return_tensors="pt")['input_ids']).last_hidden_state[:, 1:-1, :].squeeze()
                if embed.dim() > 1:
                    embed = sum(embed) / len(embed)
                embed2.append(embed)
            c = np.zeros((len(v1), len(v2)))
            for j in range(0, len(embed1)):
                for k in range(0, len(embed2)):
                    c[j, k] = cosine_similarity(embed1[j].unsqueeze(dim=0).detach().numpy(),
                                                embed2[k].unsqueeze(dim=0).detach().numpy()).min()

            while ((c.shape[0] != 0) & (c.shape[1] != 0)):
                row = np.where(c == np.max(c))[0].item()
                col = np.where(c == np.max(c))[1].item()
                sim.append(np.max(c))
                words.append([v1[row], v2[col]])
                c = np.delete(c, (row), axis=0)
                c = np.delete(c, (col), axis=1)
                del v1[row], v2[col]
        ave_score.append(sum(sim) / len(sim))
    a = ave_score.index(max(ave_score))
    tar = mult['types'].index("target")
    hard = mult['types'].index("noun")
    if a == tar:
        correct += 1
    elif a == hard:
        hard += 1
    else:
        random += 1

fast_text_vectors = api.load('fasttext-wiki-news-subwords-300')

correct = 0
hard = 0
random = 0
for mult in multiple:
    answers = mult['choices']
    source = mult['source']
    source = source.replace("’s", " is")
    source = source.replace("'t", " not")

    v1 = []
    doc = nlp(source)
    for sent in doc.sentences:
        for item in sent.words:
            if (item.upos == 'VERB')|(item.upos == 'AUX')|(item.upos == 'VB'):
                v1.append(item.text.lower())
    ave_score = []
    for answer in answers:
        answer = answer.replace("’s", " is")
        answer = answer.replace("'t", " not")

        v2 = []
        doc = nlp(answer)
        for sent in doc.sentences:
            for item in sent.words:
                if (item.upos == 'VERB') | (item.upos == 'AUX') | (item.upos == 'VB'):
                    v2.append(item.text.lower())
        doc = nlp(answer)
        sub_prop = []
        v1 = list(set(v1))
        v2 = list(set(v2))
        sim = []
        words = []
        if v1 and v2:
            for verb in v1:
                if verb in v2:
                    sim.append(1)
                    words.append([verb, verb])
                    v2.remove(verb)
            if words:
                for w in words:
                    v1.remove(w[0])
        else:
            sim.append(-100)
            words.append(['empty'])
        if v1 and v2:
            embed1 = []
            embed2 = []
            for verb in v1:
                embed = fast_text_vectors[verb]
                if embed.ndim > 1:
                    embed = sum(embed) / len(embed)
                embed1.append(embed)
            for verb in v2:
                embed = fast_text_vectors[verb]
                if embed.ndim > 1:
                    embed = sum(embed) / len(embed)
                embed2.append(embed)
            c = np.zeros((len(v1), len(v2)))
            for j in range(0, len(embed1)):
                for k in range(0, len(embed2)):
                    c[j, k] = cosine_similarity(np.expand_dims(embed1[j], axis=0),
                                                np.expand_dims(embed2[k], axis=0)).min()

            while ((c.shape[0] != 0) & (c.shape[1] != 0)):
                row = np.where(c == np.max(c))[0].item()
                col = np.where(c == np.max(c))[1].item()
                sim.append(np.max(c))
                words.append([v1[row], v2[col]])
                c = np.delete(c, (row), axis=0)
                c = np.delete(c, (col), axis=1)
                del v1[row], v2[col]
        ave_score.append(sum(sim) / len(sim))
    a = ave_score.index(max(ave_score))
    tar = mult['types'].index("target")
    hard = mult['types'].index("noun")
    if a == tar:
        correct += 1
    elif a == hard:
        hard += 1
    else:
        random += 1


tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-65b")
model = BertModel.from_pretrained("huggyllama/llama-65b")
correct = 0
hard = 0
random = 0
for mult in multiple:
    answers = mult['choices']
    source = mult['source']
    source = source.replace("’s", " is")
    source = source.replace("'t", " not")
    v1 = []
    doc = nlp(source)
    for sent in doc.sentences:
        for item in sent.words:
            if (item.upos == 'VERB')|(item.upos == 'AUX')|(item.upos == 'VB'):
                v1.append(item.text.lower())
    ave_score = []
    for answer in answers:
        answer = answer.replace("’s", " is")
        answer = answer.replace("'t", " not")
        v2 = []
        doc = nlp(answer)
        for sent in doc.sentences:
            for item in sent.words:
                if (item.upos == 'VERB') | (item.upos == 'AUX') | (item.upos == 'VB'):
                    v2.append(item.text.lower())
        doc = nlp(answer)
        sub_prop = []
        v1 = list(set(v1))
        v2 = list(set(v2))
        sim = []
        words = []
        if v1 and v2:
            for verb in v1:
                if verb in v2:
                    sim.append(1)
                    words.append([verb, verb])
                    v2.remove(verb)
            if words:
                for w in words:
                    v1.remove(w[0])
        else:
            sim.append(-100)
            words.append(['empty'])
        if v1 and v2:
            embed1 = []
            embed2 = []
            for verb in v1:
                embed = model(tokenizer(verb, return_tensors="pt")['input_ids']).last_hidden_state[:, 1:-1, :].squeeze()
                if embed.dim() > 1:
                    embed = sum(embed) / len(embed)
                embed1.append(embed)
            for verb in v2:
                embed = model(tokenizer(verb, return_tensors="pt")['input_ids']).last_hidden_state[:, 1:-1, :].squeeze()
                if embed.dim() > 1:
                    embed = sum(embed) / len(embed)
                embed2.append(embed)
            c = np.zeros((len(v1), len(v2)))
            for j in range(0, len(embed1)):
                for k in range(0, len(embed2)):
                    c[j, k] = cosine_similarity(embed1[j].unsqueeze(dim=0).detach().numpy(),
                                                embed2[k].unsqueeze(dim=0).detach().numpy()).min()

            while ((c.shape[0] != 0) & (c.shape[1] != 0)):
                row = np.where(c == np.max(c))[0].item()
                col = np.where(c == np.max(c))[1].item()
                sim.append(np.max(c))
                words.append([v1[row], v2[col]])
                c = np.delete(c, (row), axis=0)
                c = np.delete(c, (col), axis=1)
                del v1[row], v2[col]
        ave_score.append(sum(sim) / len(sim))
    a = ave_score.index(max(ave_score))
    tar = mult['types'].index("target")
    hard = mult['types'].index("noun")
    if a == tar:
        correct += 1
    elif a == hard:
        hard += 1
    else:
        random += 1


tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xxl")
model = BertModel.from_pretrained("google/flan-t5-xxl")
correct = 0
hard = 0
random = 0
for mult in multiple:
    answers = mult['choices']
    source = mult['source']
    source = source.replace("’s", " is")
    source = source.replace("'t", " not")

    v1 = []
    doc = nlp(source)
    for sent in doc.sentences:
        for item in sent.words:
            if (item.upos == 'VERB')|(item.upos == 'AUX')|(item.upos == 'VB'):
                v1.append(item.text.lower())
    ave_score = []
    for answer in answers:
        answer = answer.replace("’s", " is")
        answer = answer.replace("'t", " not")

        v2 = []
        doc = nlp(answer)
        for sent in doc.sentences:
            for item in sent.words:
                if (item.upos == 'VERB') | (item.upos == 'AUX') | (item.upos == 'VB'):
                    v2.append(item.text.lower())
        doc = nlp(answer)
        sub_prop = []
        v1 = list(set(v1))
        v2 = list(set(v2))
        sim = []
        words = []
        if v1 and v2:
            for verb in v1:
                if verb in v2:
                    sim.append(1)
                    words.append([verb, verb])
                    v2.remove(verb)
            if words:
                for w in words:
                    v1.remove(w[0])
        else:
            sim.append(-100)
            words.append(['empty'])
        if v1 and v2:
            embed1 = []
            embed2 = []
            for verb in v1:
                embed = model(tokenizer(verb, return_tensors="pt")['input_ids']).last_hidden_state[:, 1:-1, :].squeeze()
                if embed.dim() > 1:
                    embed = sum(embed) / len(embed)
                embed1.append(embed)
            for verb in v2:
                embed = model(tokenizer(verb, return_tensors="pt")['input_ids']).last_hidden_state[:, 1:-1, :].squeeze()
                if embed.dim() > 1:
                    embed = sum(embed) / len(embed)
                embed2.append(embed)
            c = np.zeros((len(v1), len(v2)))
            for j in range(0, len(embed1)):
                for k in range(0, len(embed2)):
                    c[j, k] = cosine_similarity(embed1[j].unsqueeze(dim=0).detach().numpy(),
                                                embed2[k].unsqueeze(dim=0).detach().numpy()).min()

            while ((c.shape[0] != 0) & (c.shape[1] != 0)):
                row = np.where(c == np.max(c))[0].item()
                col = np.where(c == np.max(c))[1].item()
                sim.append(np.max(c))
                words.append([v1[row], v2[col]])
                c = np.delete(c, (row), axis=0)
                c = np.delete(c, (col), axis=1)
                del v1[row], v2[col]
        ave_score.append(sum(sim) / len(sim))
    a = ave_score.index(max(ave_score))
    tar = mult['types'].index("target")
    hard = mult['types'].index("noun")
    if a == tar:
        correct += 1
    elif a == hard:
        hard += 1
    else:
        random += 1
