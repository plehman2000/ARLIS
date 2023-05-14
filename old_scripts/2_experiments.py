
import topic_extraction_methods

from topic_extraction_methods import get_topics, to_BERT_embedding,get_vector_similarity
from fuzzywuzzy import process
import transformers
# # ? HOW CAN WE ACCURATELY MEASURE THE CORRECTNESS OF TOPIC EXTRACTION METHODS?

# DATASET:
# BASE ARTICLE -> SUMMARY

# TOPIC EXTRACTION METHODS
# BASE ARTICLE -> TOPIC LIST

# VALIDATION
import numpy as np
# #? Should this take perplexity into account?

# 1) Semantic Search for 
from datasets import load_dataset

dataset = load_dataset("xsum")

train = dataset['train']

from tqdm import tqdm

samples = []
for i, sample in enumerate(tqdm(train)):
    samples.append(
        {
        "document": sample['document'],
         "summary": sample['summary']
         })
    # if i > 200:
    #     break

from fuzzywuzzy import fuzz


def get_highest_fuzzy_match_score(document, summary):
    topics = get_topics(document, method="spacy")
    # print(topics)
    top_score = 0.0
    if topics:
        matches = process.extract(summary.lower(), topics, limit=1, scorer=fuzz.partial_ratio)
        # print(summary, topics, matches)
        top_score = matches[0][1]
    #? This top score represents the best similarity between any of the topics and the summary
    return top_score


def get_highest_semantic_match_score(document, summary):
    topics = get_topics(document, method="spacy")
    # print(topics)
    topic_embeddings = []
    top_score = 0.0

    summary_embedding = to_BERT_embedding(summary)
    if topics:
        for x in topics:
            topic_embeddings.append(to_BERT_embedding(x))

        for topic_embedding in topic_embeddings:
            similarity =get_vector_similarity(topic_embedding, summary_embedding)
            if similarity > top_score:
                top_score = similarity
    #? This top score represents the best similarity between any of the topics and the summary
    return top_score
#* FUZZY Search

scores = []
for i, sample in enumerate(samples):
    document = sample['document']
    summary = sample['summary']
    score = get_highest_semantic_match_score(document, summary)
    # print(summary)
    scores.append(score)
    print(f"Running Average: {np.mean(scores)}  {i+1}/{len(samples)}", end="\r")
print(np.mean(scores))
#Compare score of extracted topics to Ground truth, which is


#get ratio of keywords to summaries
#compare to average