
from topic_extraction_methods import get_topics
import topic_extraction_methods
from fuzzywuzzy import process

# # ? HOW CAN WE ACCURATELY MEASURE THE CORRECTNESS OF TOPIC EXTRACTION METHODS?

# DATASET:
# BASE ARTICLE -> SUMMARY

# TOPIC EXTRACTION METHODS
# BASE ARTICLE -> TOPIC LIST

# VALIDATION

# #? Should this take perplexity into account?

# 1) Semantic Search for 
from datasets import load_dataset

dataset = load_dataset("xsum")

train = dataset['train']

samples = []
for i, sample in enumerate(train):
    samples.append(
        {
        "document": sample['document'],
         "summary": sample['summary']
         })
    if i > 20:
        break





def get_highest_fuzzy_match_score(document, summary):
    topics = get_topics(document)
    print(topics)
    matches = process.extract(summary.lower(), topics, limit=1)
    # print(summary, topics, matches)
    top_score = matches[0][1]
    #? This top score represents the best similarity between any of the topics and the summary
    return top_score
#* FUZZY Search

for sample in samples:
    document = sample['document']
    summary = sample['summary']
    score = get_highest_fuzzy_match_score(document, summary)
    print(summary)
    print(score)

#Compare score of extracted topics to Ground truth, which is