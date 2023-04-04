from itertools import combinations, product
import multiprocessing
from datetime import datetime

import pandas as pd
from sklearn.metrics import adjusted_rand_score
import networkx as nx


def ngram_overlap(data_dict, overlap):
    '''Returns edges between articles given overlap threshold.'''

    global data

    data = data_dict

    # Compare pairs of passages, calculate overlap percentage and keep pair if meets overlap threshold
    print("\n Calculating overlaps ...")
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)
    list_of_edges_lists = pool.starmap(compare_passage_pairs, [(date, overlap) for date in list(data.keys())])
    pool.close()

    # Collapse into single list
    edges_list = [item for sublist in list_of_edges_lists for item in sublist]
    return edges_list


def compare_passage_pairs(date_str, overlap):
    """Module for calculating N-Gram overlap on multiple cores"""

    same_day_dat = data[date_str]
    n_same_day = len(same_day_dat['art_list'])
    edges = []

    # iterate through all other dates
    for alt_date in list(data.keys()):
        date_dt = datetime.strptime(date_str, "%b-%d-%Y")
        new_dt = datetime.strptime(alt_date, "%b-%d-%Y")

        # only compare to later dates (to prevent repetitions)
        if date_dt >= new_dt:
            new_day_dat = data[alt_date]
            n_new_day = len(new_day_dat['art_list'])

            print(f"\n Computing N-gram overlap between {date_str} and {alt_date} passages...")
            for i, j in product(range(n_same_day), range(n_new_day)):
                compare_overlap_to_threshold(i, j, data_i=same_day_dat, data_j=new_day_dat, overlap=overlap, outfile=edges)

    return edges


def compare_overlap_to_threshold(i, j, data_i, data_j, overlap, outfile):

    # count ngram overlaps
    passage_i_set = data_i['ngram_list'][i]
    passage_j_set = data_j['ngram_list'][j]
    intersect = passage_i_set.intersection(passage_j_set)
    overlap_count = len(intersect)

    # compute percentage of possible ngrams that overlapped
    if len(passage_i_set) != 0 and len(passage_j_set) != 0:
        # overlap_pct = overlap_count / min(len(passage_i_set), len(passage_j_set))
        overlap_pct = overlap_count / len(passage_i_set.union(passage_j_set))
    else:
        overlap_pct = 0

    # compare to overlap threshold and add edge if meets threshold
    if overlap_pct >= overlap:
        id_i = data_i['id_list'][i]
        id_j = data_j['id_list'][j]
        text_i = data_i['art_list'][i]
        text_j = data_j['art_list'][j]
        outfile.append((
            id_i,
            id_j,
            {
                'text_1': text_i,
                'text_2': text_j,
                'overlap': overlap_pct
            }
        ))


def clusters_from_edges(edges_list):
    """Identify clusters of passages given a dictionary of edges"""

    # clusters via NetworkX
    G = nx.Graph()
    G.add_edges_from(edges_list)
    sub_graphs = [G.subgraph(c).copy() for c in nx.connected_components(G)]

    sub_graph_dict = {}
    for i in range(len(sub_graphs)):
        sub_graph_dict[i] = list(sub_graphs[i].nodes())

    return sub_graph_dict


def evaluate(pred_pairs, gt_pairs, all_ids):
    
    # Evaluate
    set_preds = set(map(tuple, pred_pairs))
    set_gt = set(map(tuple, gt_pairs))

    # Metrics
    true_pos = [i for i in set_gt if i in set_preds or (i[1], i[0]) in set_preds]
    false_pos = [i for i in set_preds if i not in set_gt and (i[1], i[0]) not in set_gt]
    false_neg = [i for i in set_gt if i not in set_preds and (i[1], i[0]) not in set_preds]

    tps = len(true_pos)
    fps = len(false_pos)
    fns = len(false_neg)

    precision = tps / (tps + fps)
    recall = tps / (tps + fns)
    f_score = 2 * (precision * recall) / (precision + recall)

    print(precision, recall, f_score)
    print(tps, fps, fns)
    
    # ARI 
    pred_clusters = clusters_from_edges(set_preds)
    gt_clusters = clusters_from_edges(set_gt)

    # get dictionary mapping article to cluster number
    pred_dict = {}
    pred_count = 0
    for cluster in pred_clusters:
        for article in pred_clusters[cluster]:
            pred_dict[article] = pred_count
        pred_count += 1

    gt_dict = {}
    gt_count = 0
    for cluster in gt_clusters:
        for article in gt_clusters[cluster]:
            gt_dict[article] = gt_count
        gt_count += 1

    # fill in clusters with unclustered articles
    full_pred_clusters = []
    full_gt_clusters = []
    for article in all_ids:
        if article in pred_dict:
            full_pred_clusters.append(pred_dict[article])
        else:
            full_pred_clusters.append(pred_count)
            pred_count += 1

        if article in gt_dict:
            full_gt_clusters.append(gt_dict[article])
        else:
            full_gt_clusters.append(gt_count)
            gt_count += 1

    assert len(full_pred_clusters) == len(full_gt_clusters)

    ARI = adjusted_rand_score(full_pred_clusters, full_gt_clusters)

    print("ARI:", ARI)

    return ARI


def make_jaccard_preds(text_dict, overlap, n):

    ## Get pairs with n-gram overlap
    # Create list of n-grams for each article
    clean_texts = remove_odd_characters(list(text_dict.values()))
    clean_texts = [t.lower() for t in clean_texts]

    # clean_texts = list(text_dict.values())
    
    n_gram_list = list_ngrams(clean_texts, n_gram_size=n)
    data_dict = {"Oct-03-1968": {
        "id_list": list(text_dict.keys()),
        "art_list": clean_texts,
        "ngram_list": n_gram_list
    }}

    # Calculate n-gram overlaps and return pairs that meet overlap threshold
    pairs_to_compare = ngram_overlap(data_dict, overlap=overlap)

    # Remove ones that are from the same scan
    pairs_to_compare = [p for p in pairs_to_compare if "_".join(p[0].split("_")[1:]) != "_".join(p[1].split("_")[1:])]

    # Remove duplicates
    no_dup_pairs_to_compare = []
    check_list = []
    for p in pairs_to_compare:
        if [p[0], p[1]] not in check_list and [p[1], p[0]] not in check_list:
            no_dup_pairs_to_compare.append(p)
            check_list.append([p[0], p[1]])
    pairs_to_compare = no_dup_pairs_to_compare

    pairs_to_compare = [list(p[:2]) for p in pairs_to_compare]

    return pairs_to_compare


def list_ngrams(list_of_texts, n_gram_size=5, concat=False, char=False):
    ''' Returns list of n-grams given list of texts. '''

    # Create list of all n-grams in all passages
    ngram_sets = []
    for passage in list_of_texts:
        # Creates character-based n-grams
        if char:
            words = passage.split()
            passage = " ".join(words)
            n_grams = list(zip(*[passage[i:] for i in range(n_gram_size)]))
        # Creates word-based n-grams
        else:
            words = passage.split()
            n_grams = list(zip(*[words[i:] for i in range(n_gram_size)]))
        
        # concatenates n-grams instead of leaving them as tuples
        if concat:
            n_grams = [" ".join(x) for x in n_grams]

        ngram_sets.append(set(n_grams))

    return ngram_sets


def remove_odd_characters(list_of_texts):
    ''' Removes punctuation, unknown characters. '''
    chars_to_remove = r'"#$%&\()*+/:;<=>@[\\]^_`{|}~.?,!\''
    ocr_article_clean_texts = []

    for text in list_of_texts:
        text = text.replace("-\n", "").replace("\n", " ")
        text = text.translate(str.maketrans('', '', chars_to_remove))
        text = text.encode('ascii', 'ignore').decode()
        ocr_article_clean_texts.append(text)

    return ocr_article_clean_texts


def open_data(path):

    df = pd.read_csv(path)

    images_paths = list(df['image_path'])
    texts = list(df['text'])
    labels = list(df['label'])

    js = []
    for i, path in enumerate(images_paths):
        js.append(
            {
            "image_path": path,
            "text": str(texts[i]),
            "label": labels[i]
            }
        )

    cluster_dict = {}
    text_dict = {}
    
    for cap in js:
        text_dict[cap["image_path"]] = cap["text"]

        if cap["label"] not in cluster_dict:
            cluster_dict[cap["label"]] = []
        cluster_dict[cap["label"]].append(cap["image_path"])

    gt_pairs = []
    for clu in cluster_dict:
        edges = combinations(cluster_dict[clu], 2)
        edges = [list(e) for e in edges]
        gt_pairs.extend(edges)

    return gt_pairs, text_dict


def find_best_hyperparameters(val_data_path):

    val_gt_pairs, val_text_dict = open_data(path=val_data_path)
    ids = list(val_text_dict.keys())

    best_ari = 0
    best_n = 0
    best_ol = 0
    for n in [1,2,3,4,5]:
        for ol in [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]:
            
            pred_pairs = make_jaccard_preds(val_text_dict, overlap=ol, n=n)
            ari = evaluate(gt_pairs=val_gt_pairs, pred_pairs=pred_pairs, all_ids=ids)

            if ari > best_ari:
                best_ari = ari
                best_n = n
                best_ol = ol
                
    return best_n, best_ol


if __name__ == '__main__':

    # Finetune on val data
    best_n, best_ol = find_best_hyperparameters(val_data_path='path/to/val/data')

    # Run with best hyperparameters on test data 
    gt_pairs, text_dict = open_data(path = 'path/to/test/data')

    pred_pairs = make_jaccard_preds(text_dict, overlap=best_ol, n=best_n)

    evaluate(pred_pairs=pred_pairs, gt_pairs=gt_pairs, all_ids=list(text_dict.keys()))