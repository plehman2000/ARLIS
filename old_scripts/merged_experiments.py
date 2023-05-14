
from topic_extraction_methods import get_topics, get_topics_spacy
import topic_extraction_methods
from fuzzywuzzy import process

# # ? HOW CAN WE ACCURATELY MEASURE THE CORRECTNESS OF TOPIC EXTRACTION METHODS?

# DATASET:
# BASE ARTICLE -> SUMMARY

# TOPIC EXTRACTION METHODS
# BASE ARTICLE -> TOPIC LIST

# VALIDATION
import numpy as np
# #? Should this take perplexity into account?



from tqdm import tqdm

import pandas as pd

df = pd.read_csv("Merged_Annotations.csv")

#! Compare results on manual data
x = df
# strip entries hat are short
#! do five nume sum
    # if i > 200:
    #     break

# from fuzzywuzzy import fuzz


# def get_highest_fuzzy_match_score(document, summary):
#     topics = get_topics_spacy(document)
#     # print(topics)
#     top_score = 0.0
#     if topics:
#         matches = process.extract(summary.lower(), topics, limit=1, scorer=fuzz.partial_ratio)
#         # print(summary, topics, matches)
#         top_score = matches[0][1]
#     #? This top score represents the best similarity between any of the topics and the summary
#     return top_score
# #* FUZZY Search

# scores = []
# for i, sample in enumerate(samples):
#     document = sample['document']
#     summary = sample['summary']
#     score = get_highest_fuzzy_match_score(document, summary)
#     # print(summary)
#     scores.append(score)
#     print(f"Running Average: {np.mean(scores)}  {i+1}/{len(samples)}", end="\r")
# print(np.mean(scores))
# #Compare score of extracted topics to Ground truth, which is


# #get ratio of keywords to summaries
# #compare to average