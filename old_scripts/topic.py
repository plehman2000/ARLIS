import os
from datetime import date
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import strip_tags
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
# import umap
import numpy as np
from numpy.linalg import norm
from sklearn.feature_extraction.text import CountVectorizer
import spacy
import nltk
from nltk import tokenize
import requests
import wikipedia
import wikipediaapi
import traceback2

wiki_wiki = wikipediaapi.Wikipedia('en')

nltk.download('punkt')
nlp = spacy.load('en_core_web_lg')



def collect_data(df, text_column):
    """Filters data limiting to text column only.
    Keyword arguments:
    :param df: A dataframe with all the input data
    :param text_column: column name of the text column from the input datastructure that
    :return: Both a list of text column data and the dataframe itself
    """
    try:
        dataset = df
        print(dataset.shape)

        data = dataset[text_column].tolist()
        dataset.head()
    except:
        print('EOF')
        pass
    return data, dataset


def tokenizer(document):
    """Converts document/data into strings of tokens.
    Keyword arguments:
    :param document: List with all the input data.
    :return: A function call to a predefined gensim function that preprocesses data
    """
    return simple_preprocess(strip_tags(document), deacc=True)


def find_embeddings_using_transformers(data):
    """Creates embeddings for each token in the list of tokens parameter using sentence_transformers .
    Keyword arguments:
    :param data: List of tokens retrieved from input document after preprocessing
    :return: embeddings for each token
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-mpnet-base-v2')
    embeddings = model.encode(data, show_progress_bar=False)
    return embeddings


def dimentionality_reduction_using_umap(embeddings):
    """Reduce dimensions using umap.
    Keyword arguments:
    :param embeddings: embeddings received from transformers
    :return: embeddings after dimentionality reduction
    """
    import umap
    umap_embeddings = umap.UMAP(n_neighbors=15, 
                                n_components=5, 
                                metric='cosine').fit_transform(embeddings)
    return umap_embeddings


def normalize_vectors(umap_embeddings):
    """Normalize vector embeddings.
    Keyword arguments:
    :param umap_embeddings: embeddings post dimensionality reduction
    :return: Normalized vectors
    """
    length = np.sqrt((umap_embeddings**2).sum(axis=1))[:,None]
    umap_embeddings = umap_embeddings / length
    return umap_embeddings


def perform_clustering(normalized_embeddings, num_c = 25):
    """Performs clustering on the documents using KMeans.
    Keyword arguments:
    :param normalized_embeddings: normalized embeddings
    :param num_c: number of clusters to form (default 25)
    :return: Cluster, KMeans object and normalized vector embeddings
    """
    km = KMeans(n_clusters=num_c, init='k-means++', n_init=20)

    print("Clustering with %s" % km)
    cluster = km.fit(normalized_embeddings)
    return cluster, km, normalized_embeddings


def single_document_per_topic(data, dataset, cluster, doc_id):
    """Groups documents based on the topic using the clusters formed in the above method .
    Keyword arguments:
    :param data: list of tokens extracted from documents
    :param dataset: text column data only
    :param cluster: clusters obtained from Kmeans clustering
    :param doc_id: document id column from the input document set/data
    :return: document dataframe and docs per topic
    """
    docs_df = pd.DataFrame(data, columns=["Doc"])
    
    docs_df[doc_id] = dataset[doc_id].tolist()
    docs_df['Topic'] = cluster.labels_
    docs_df['Doc_ID'] = range(len(docs_df))
    docs_per_topic = docs_df.groupby(['Topic'], as_index = False).agg({'Doc': ' '.join, doc_id:lambda x: list(x)})
    
    return docs_df, docs_per_topic


def c_tf_idf(documents, data_len, ngram_range=(1,3)):
    """Returns tf-idf vectors from the input documents.
    Keyword arguments:
    :param documents: documents dataframe with topics, doc-ids
    :param data_len: length of tokenized data list
    :param ngram_range: range of ngrams or consecutive words (default (1,3))
    :return: tf_idf value and instance of CountVectorizer
    """
    count = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(documents)
    t = count.transform(documents).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(data_len, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)

    return tf_idf, count


def extract_top_x_words_per_topic(tf_idf, count, docs_per_topic, num_words=20):
    """Extracts top x words from the input documents using tf-idf.
    Keyword arguments:
    :param tf_idf: tf-idf vectors
    :param count: CountVectorizer model
    :param docs_per_topic: grouped documents by topic
    :param num_words: number of top words to be extracted (default 20)
    :return: top x words from input documents
    """
    words = count.get_feature_names_out()
    labels = list(docs_per_topic.Topic)
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -num_words:]
    top_x_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}
    return top_x_words


def extract_topic_sizes(df):
    """Returns number of topics with their corresponding document ids grouped through clustering
    Keyword arguments:
    :param df: dataframe with topics and ids
    :return: count of docs grouped as one for each topic
    """
    topic_sizes = (df.groupby(['Topic'])
                     .Doc
                     .count()
                     .reset_index()
                     .rename({"Topic": "Topic", "Doc": "Size"}, axis='columns')
                     .sort_values("Size", ascending=False))
    return topic_sizes


def get_summary(title):
    """Queries Wikipedia API and returns wikipedia pages based on search titles.
    Keyword arguments:
    :param title: ngram search keywords for the wikipedia summary API
    :return: Wikipedia pages content based on search query
    """
    URL = "https://en.wikipedia.org/w/api.php"
    S = requests.Session()

    PARAMS = {
        "action":"query",
        "format":"json",
        "prop":"extracts",
        "generator":"prefixsearch",
        "redirects":"1",
        "converttitles":"1",
        "formatversion":"2",
        "exintro":"1",
        "explaintext":"1",
        "gpssearch":title
    }

    R = S.get(url=URL, params=PARAMS)
    DATA = R.json()
    
    if 'query' in DATA:
        PAGES = DATA['query']['pages']

        return PAGES
    else:
        return []


def wiki_docs(top_x_words):
    """Prepares candidate documents for each ngram in top_x_words input using get_summary method
    Keyword arguments:
    :param top_x_words: top x words exracted from documents using tf-idf
    :return: candidate wikipedia documents found based on top_x_words search criteria
    """  
    candidate_dict = {}
    sim_dict=dict()
    title_lst=[]

    run_count = 0
    exception_count = 0

    for i in range(len(top_x_words)):
        documents_dict = dict()
        key_words = []

        for each_gram, gram_score in top_x_words[i]:
            key_words.append(each_gram)

    #         get the topics for each gram
    #         for topic_text in get_topics(each_gram):
            pages = get_summary(each_gram)
            for page in pages:
                try:
                    run_count += 1
                    if (page['title'] not in documents_dict.keys()):
                        text = page['extract']
                        text = text.replace('\n', '. ')
                        documents_dict[page['title']] = text
                except Exception as e:
                    print(e)
                    exception_count += 1
                    pass

        candidate_dict[i] = documents_dict

        print("Number of documents found", len(documents_dict))


    print(f"The code incurred exception {exception_count*100/run_count}%")
    return candidate_dict, documents_dict, text


def output_format(output_dict):
    """Prepares output dataframe in the required format
    Keyword arguments:
    :param output_dict: dictionary with doc ids, topics, similarity scores and word clusters
    :return: formatted output dataframe
    """  

    output_list = []
    for doc_id, result in output_dict.items():
        result_list = list(result.items())
        output_list.append(([f'{doc_id}'], result_list[0][0], result_list[0][1], result_list[1][1]))
    outputDF = pd.DataFrame(output_list, columns=["Documents", "Topics", "Similarity_Score", "Topic_Cluster"])
    return outputDF


def output_format_single_doc(output_dict):
    """Prepares output dataframe in the required format when a single document is passed
    Keyword arguments:
    :param output_dict: dictionary with doc ids, topics, similarity scores and word clusters
    :return: formatted output dataframe when a single doc is passed as input
    """
    doc_id = list(output_dict)[0]
    keys = list(list(output_dict.values())[0].keys())
    values = list(list(output_dict.values())[0].values())
    out_list = []
    for i in range(5):
        out_list.append(([f'{doc_id}'], keys[i], values[i], values[-1]))
        outputDF = pd.DataFrame(out_list, columns=["Documents", "Topics", "Similarity_Score", "Topic_Cluster"])
    return outputDF


def get_sentences(text, count=3):
    """Prepares first three tokenized sentences of wikipedia documents that match our topic words
    Keyword arguments:
    :param text: text from matched wikipedia documents
    :param count: number of sentences to pick from the wikipedia document to pass to tf-idf vectorizer to compute similarity (default 3)
    :return: tokenized sentences
    """
    return text[0] + " " + " ".join(tokenize.sent_tokenize(text[1])[:count])


def most_similar_using_td_idf(cand_dict, top_x_words, docs_per_topic, docs_df, doc_id):
    """Prepares output dataframe with document ids, topics, similarity scores uiang tf-idf
    Keyword arguments:
    :param cand_dict: candidate dictionary containing documents/pages queried from wikipedia
    :param top_x_words: top x words exracted from documents using tf-idf
    :param docs_per_topic: grouped documents by topic
    :param docs_df: dataframe with doc ids and topics
    :param doc_id: document id column of the input document set/data
    :return: an output list and output dataframe with topics and their similarity scores with matched wikipedia documents 
    """

    output = {}
    for i in range(len(top_x_words)):
        vect = TfidfVectorizer(ngram_range=(1,2))
        documents_dict = cand_dict[i]
        doc_vector = vect.fit_transform(map(get_sentences,documents_dict.items())).toarray()
        key_words_vector = vect.transform([docs_per_topic['Doc'][i]]).toarray().reshape((-1,))

        dot_product = np.dot(doc_vector, key_words_vector)
        
        output[docs_df.loc[docs_df['Topic']==i][doc_id].iloc[0]] = {list(documents_dict.keys())[np.argmax(dot_product)] : max(dot_product)}
        output[docs_df.loc[docs_df['Topic']==i][doc_id].iloc[0]].update(
        {"Trigrams": [word for word, score in top_x_words[i]]})
        outputDF = output_format(output)
        outputDF['Documents'] = docs_per_topic[doc_id]
        outputDF_sorted = outputDF.sort_values(by=["Similarity_Score"], ascending=False)
        
        
    return output, outputDF_sorted


def save_results(file_name, outputDf, file_path):
    """Saves final datafrae result into a csv file
    Keyword arguments:
    :param file_name: name of the csv file to be created
    :param outputDf: output dataframe with document ids, corresponding topics and similarity scores
    :param file_path: path on the host machine where the newly created csv file is to be stored
    """
    path = os.path.dirname(os.path.realpath('__file__'))
    if not os.path.exists(file_path):
        print("creating topic_detection_output folder to store output")
        os.makedirs(file_path)
    outputDf.to_csv(f'{path}/{file_path}/{file_name}.csv', index=False)


def most_similar_using_bert(cand_dict, top_x_words, docs_per_topic, docs_df, doc_id):
    """Prepares output dataframe with document ids, topics, similarity scores uiang bert
    Keyword arguments:
    :param cand_dict: candidate dictionary containing documents/pages queried from wikipedia
    :param top_x_words: top x words exracted from documents using tf-idf
    :param docs_per_topic: grouped documents by topic
    :param docs_df: dataframe with doc ids and topics
    :param doc_id: document id column of the input document set/data
    :return: output list and output dataframe with topics and their similarity scores with matched wikipedia documents
    """
    output = {}
    for i in range(len(top_x_words)):
        documents_dict = cand_dict[i]
        candidates_for_document = list(map(get_sentences,documents_dict.items()))
        candidate_embeddings = find_embeddings_using_transformers(candidates_for_document)
        
        doc_embedding = find_embeddings_using_transformers([docs_per_topic['Doc'][i]])
        dot_product = np.dot(candidate_embeddings, doc_embedding.reshape(-1))
        # build the output in the format as above
        output[docs_df.loc[docs_df['Topic']==i][doc_id].iloc[0]] = {list(documents_dict.keys())[np.argmax(dot_product)] : max(dot_product)}
        
        output[docs_df.loc[docs_df['Topic']==i][doc_id].iloc[0]].update(
        {"Trigrams": [word for word, score in top_x_words[i]]})
        
        outputDF = output_format(output)
        outputDF['Documents'] = docs_per_topic[doc_id]
        outputDF_sorted = outputDF.sort_values(by=['Similarity_Score'], ascending=False)

        if i%10==0:
            print(f"Finished processing {i} documents")
    return output, outputDF_sorted

def most_similar_using_bert_single_doc(cand_dict, top_x_words, docs_per_topic, docs_df, doc_id):
    """Prepares output dataframe with document ids, topics, similarity scores uiang bert when a single doc is passed as input
    Keyword arguments:
    :param cand_dict: candidate dictionary containing documents/pages queried from wikipedia
    :param top_x_words: top x words exracted from documents using tf-idf
    :param docs_per_topic: grouped documents by topic
    :param docs_df: dataframe with doc ids and topics
    :param doc_id: document id column of the input document set/data
    :return: output list and output dataframe with topics and their similarity scores with matched wikipedia documents
    """
    documents_dict = cand_dict[0]
    candidates_for_document = list(map(get_sentences,documents_dict.items()))
    candidate_embeddings = find_embeddings_using_transformers(candidates_for_document)    
    doc_embedding = find_embeddings_using_transformers([docs_per_topic['Doc'][0]])
    dot_product = np.dot(candidate_embeddings, doc_embedding.reshape(-1))
    dot_productt = dot_product.tolist()
    output={}
    res = sorted(range(len(dot_productt)), key=lambda i: dot_productt[i], reverse=True)[:5]
    for i in range(5):
        if(i==0):
                output[docs_df.loc[docs_df['Topic']==i][doc_id].iloc[0]] = {list(documents_dict.keys())[res[i]] : dot_productt[res[i]]}
        elif(i==4):
                output[list(docs_per_topic[doc_id][0])[0]].update({list(documents_dict.keys())[res[i]] : dot_productt[res[i]]})
                output[list(docs_per_topic[doc_id][0])[0]].update({"Trigrams": [word for word, score in top_x_words[0]]})
        else:
                output[list(docs_per_topic[doc_id][0])[0]].update({list(documents_dict.keys())[res[i]] : dot_productt[res[i]]})
    
    outputDF = output_format_single_doc(output)
    outputDF['Documents'] = docs_per_topic[doc_id]
    outputDF_sorted = outputDF.sort_values(by=['Similarity_Score'], ascending=False)
    return output, outputDF_sorted


def topic_modeling(json, num_of_clusters=25, filename=date.today()):
    """Main method that is called from the terminal after importing the topic module. Makes calls to all other methods involved in topic detection.
    Keyword arguments:
    :param json: input list of jsons. Each json in the list must follow the pattern {"doc_id": __, "info_source_id":__, "doc_text": "some text",        "date_published":"YYYY-MM-DDTHH:MM:SS.SSZ"}
    :param num_of_clusters: number of clusters to be formed (default 25)
    :param filename: name of the output csv file (default current date)
    :return: final output dataframe with document ids, topics, corresponding similarity scores and word clusters used to produce those topics
    """
    df = pd.DataFrame(json)
    print(df)
    df_columns = list(df.columns)
    doc_id = df_columns[0]
    text_column = df_columns[2]
    data, dataset = collect_data(df, text_column)
    if(len(data)<25):
        num_of_clusters = len(data)
    data = [' '.join(tokenizer(str(doc))) for doc in data]
    # find the actual labels from the dataset, this will be later used in evaluation
    embeddings = find_embeddings_using_transformers(data)
    cluster, km, embeddings = perform_clustering(normalize_vectors(embeddings), num_of_clusters)
    
    ngram_range = (1,3)
    docs_df, docs_per_topic = single_document_per_topic(data, dataset, cluster, doc_id)
    tf_idf, count = c_tf_idf(docs_per_topic.Doc.values, len(data), ngram_range)

    top_x_words = extract_top_x_words_per_topic(tf_idf, count, docs_per_topic, num_words=20)
    topic_sizes = extract_topic_sizes(docs_df); topic_sizes.head(100)
    
    file_path = "topic_detection_output"
    
    candidate_dict, documents_dict, text = wiki_docs(top_x_words)
    
    suffix = "Trigrams"
    sentence_count = 3

    # output_tfidf, outputDF_tfidf = most_similar_using_td_idf(candidate_dict, top_x_words, docs_per_topic, docs_df, doc_id)
    # file_name = f"{filename}_topics"
    # save_results(file_name, output1, outputDF1, file_path)
    
    if(len(candidate_dict)==1):
        output_bert, outputDF_bert = most_similar_using_bert_single_doc(candidate_dict, top_x_words, docs_per_topic, docs_df, doc_id)
    else:
        output_bert, outputDF_bert = most_similar_using_bert(candidate_dict, top_x_words, docs_per_topic, docs_df, doc_id)
    file_name = f"{filename}_topics"
    save_results(file_name, outputDF_bert, file_path)
    return outputDF_bert