import numpy as np
import pandas as pd
dataset = pd.read_csv("StoryAnalogy.csv")
import stanza
from grakel import Graph as GrakelGraph
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from zss import simple_distance, Node as ZSSNode
stanza.download('en')
import pickle
nlp = stanza.Pipeline('en', processors='tokenize,pos,constituency')
import re
from transformers import AutoTokenizer, BertModel
import torch
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = BertModel.from_pretrained("google-bert/bert-base-uncased")
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
doc = nlp('He carefully strikes a single match and places it against the kindling')
for sentence in doc.sentences:
    print(sentence.constituency)

labels1 = []
edges1 = []
children1 = []
labels2 = []
edges2 = []
children2 = []
counter = 0
def tree_traverse(tree, parent = None):

        global counter
        if type(tree) == stanza.models.constituency.parse_tree.Tree:
            labels[counter]= tree.label


        if (parent !=None) & (tree.children != 0):

            edges.append((parent, counter))
            children.setdefault(parent, []).append(counter)

        parent = counter
        counter += 1
        if type(tree) == stanza.models.constituency.parse_tree.Tree:
            for child in tree.children:
                tree_traverse(child, parent)

        return 0


tree_traverse(doc.sentences[0].constituency)

def extract_from_stanza_tree(stanza_tree):
    labels = {}
    edges = []
    children_map = {}
    node_id_counter = [0]
    node_map = {}

    def traverse(node, parent_id=None):
        node_id = node_id_counter[0]
        node_id_counter[0] += 1

        labels[node_id] = node.label
        node_map[id(node)] = node_id

        if parent_id is not None:
            edges.append((parent_id, node_id))
            children_map.setdefault(parent_id, []).append(node_id)

        for child in node.children:
            # Stanza terminal nodes are just strings (words), not ConstituencyTree objects
            if isinstance(child, str):
                leaf_id = node_id_counter[0]
                node_id_counter[0] += 1
                labels[leaf_id] = child
                edges.append((node_id, leaf_id))
                children_map.setdefault(node_id, []).append(leaf_id)
            else:
                traverse(child, node_id)

    traverse(stanza_tree)
    return labels, edges, children_map


l1 = []
l2 = []
for i in range(0,len(dataset)):
    l1.append(dataset['s1'][i].count('.')+dataset['s1'][i].count('?')+dataset['s1'][i].count('!'))
    l2.append(dataset['s2'][i].count('.')+dataset['s2'][i].count('?')+dataset['s2'][i].count('!'))

z = [(x == y) for x, y in zip(l1, l2)]
dataset_same_all = dataset[z].reset_index(drop=True)
prop = []
for i in range(0,len(dataset_same_all)):
    s1 = list(filter(None,re.split(r'[.?!]',dataset_same_all['s1'][i].replace("\"",""))))
    s2 = list(filter(None,re.split(r'[.?!]',dataset_same_all['s2'][i].replace("\"",""))))
    sub_prop = []
    for j in range(0,len(s1)):
        doc1 = nlp(s1[j])
        label1, edge1, child1 = extract_from_stanza_tree(doc1.sentences[0].constituency)
        doc2 = nlp(s2[j])
        label2, edge2, child2 = extract_from_stanza_tree(doc2.sentences[0].constituency)
        tree1 = GrakelGraph(edge1, node_labels=label1)
        tree2 = GrakelGraph(edge2, node_labels=label2)
        wl_kernel = WeisfeilerLehman(n_iter=3)
        K = wl_kernel.fit_transform([tree1, tree2])
        sub_prop.append((K[0, 1] / K[0, 0] + K[1, 0] / K[1, 1])/2)
    prop.append(sum(sub_prop) / len(sub_prop))

with open('prop.pkl', 'wb') as f:  # open a text file
    pickle.dump(prop, f)
f.close()

dataset_same_all['s1'][2630] = dataset_same_all['s1'][2630].replace('lower','lowers')


score = []
matches = []
ave_score = []
nlp = stanza.Pipeline('en', processors='tokenize,pos')
for i in range(0,len(dataset_same_all)):
    v1 = []
    v2 = []
    doc = nlp(dataset_same_all['s1'][i])
    for sent in doc.sentences:
        for item in sent.words:
            if (item.upos == 'VERB')|(item.upos == 'AUX')|(item.upos == 'VB'):
                v1.append(item.text.lower())
    doc = nlp(dataset_same_all['s2'][i])
    for sent in doc.sentences:
        for item in sent.words:
            if (item.upos == 'VERB')|(item.upos == 'AUX')|(item.upos == 'VB'):
                v2.append(item.text.lower())
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
            embed = model(tokenizer(verb, return_tensors="pt")['input_ids']).last_hidden_state[:,1:-1,:].squeeze()
            if embed.dim()>1:
                embed = sum(embed)/len(embed)
            embed1.append(embed)
        for verb in v2:
            embed = model(tokenizer(verb, return_tensors="pt")['input_ids']).last_hidden_state[:,1:-1,:].squeeze()
            if embed.dim() >1:
                embed = sum(embed)/len(embed)
            embed2.append(embed)
        c = np.zeros((len(v1), len(v2)))
        for j in range(0,len(embed1)):
            for k in range(0,len(embed2)):
                c[j,k] = cosine_similarity(embed1[j].unsqueeze(dim=0).detach().numpy(),embed2[k].unsqueeze(dim=0).detach().numpy()).min()

        while((c.shape[0] != 0) & (c.shape[1] != 0)):
            row =np.where(c==np.max(c))[0].item()
            col =np.where(c==np.max(c))[1].item()
            sim.append(np.max(c))
            words.append([v1[row], v2[col]])
            c = np.delete(c, (row), axis=0)
            c = np.delete(c, (col), axis=1)
            del v1[row], v2[col]
    score.append(sim)
    matches.append(words)
    ave_score.append(sum(sim)/len(sim))

with open('score.pkl', 'wb') as f:  # open a text file
    pickle.dump(score, f)
f.close()
with open('matches.pkl', 'wb') as f:  # open a text file
    pickle.dump(matches, f)
f.close()
with open('ave_score.pkl', 'wb') as f:  # open a text file
    pickle.dump(ave_score, f)
f.close()

stats.spearmanr(ave_score, dataset_same_all['relation'])


score_noun = []
matches_noun = []
ave_score_noun = []
nlp = stanza.Pipeline('en', processors='tokenize,pos')
for i in range(0,len(dataset_same_all)):
    v1 = []
    v2 = []
    doc = nlp(dataset_same_all['s1'][i])
    for sent in doc.sentences:
        for item in sent.words:
            if (item.upos == 'NOUN'):
                v1.append(item.text.lower())
    doc = nlp(dataset_same_all['s2'][i])
    for sent in doc.sentences:
        for item in sent.words:
            if (item.upos == 'NOUN'):
                v2.append(item.text.lower())
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
            embed = model(tokenizer(verb, return_tensors="pt")['input_ids']).last_hidden_state[:,1:-1,:].squeeze()
            if embed.dim()>1:
                embed = sum(embed)/len(embed)
            embed1.append(embed)
        for verb in v2:
            embed = model(tokenizer(verb, return_tensors="pt")['input_ids']).last_hidden_state[:,1:-1,:].squeeze()
            if embed.dim() >1:
                embed = sum(embed)/len(embed)
            embed2.append(embed)
        c = np.zeros((len(v1), len(v2)))
        for j in range(0,len(embed1)):
            for k in range(0,len(embed2)):
                c[j,k] = cosine_similarity(embed1[j].unsqueeze(dim=0).detach().numpy(),embed2[k].unsqueeze(dim=0).detach().numpy()).min()

        while((c.shape[0] != 0) & (c.shape[1] != 0)):
            row =np.where(c==np.max(c))[0].item()
            col =np.where(c==np.max(c))[1].item()
            sim.append(np.max(c))
            words.append([v1[row], v2[col]])
            c = np.delete(c, (row), axis=0)
            c = np.delete(c, (col), axis=1)
            del v1[row], v2[col]
    score_noun.append(sim)
    matches_noun.append(words)
    ave_score_noun.append(sum(sim)/len(sim))

with open('score_noun.pkl', 'wb') as f:  # open a text file
    pickle.dump(score_noun, f)
f.close()
with open('matches_noun.pkl', 'wb') as f:  # open a text file
    pickle.dump(matches_noun, f)
f.close()
with open('ave_score_noun.pkl', 'wb') as f:  # open a text file
    pickle.dump(ave_score_noun, f)
f.close()

stats.spearmanr(ave_score, dataset_same_all['relation'])