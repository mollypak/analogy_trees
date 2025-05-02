import pandas as pd
dataset = pd.read_csv("StoryAnalogy.csv")
import stanza
from grakel import Graph as GrakelGraph
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from zss import simple_distance, Node as ZSSNode
stanza.download('en')
import pickle
nlp = stanza.Pipeline('en', processors='tokenize,pos,constituency')


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
    s1 = list(filter(None,dataset_same_all['s1'][i].replace("\"","").split('.')))
    s2 = list(filter(None,dataset_same_all['s2'][i].replace("\"","").split('.')))
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




