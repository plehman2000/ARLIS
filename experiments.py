# import topic_extraction_methods

# from topic_extraction_methods import get_topics, get_topics_spacy, to_BERT_embedding,get_vector_similarity
# from fuzzywuzzy import process
# import transformers
# import numpy as np

# # 1) Semantic Search for 
# from datasets import load_dataset
import numpy as np

import pandas as pd

df = pd.read_csv("fort.csv")
# print(df)
samples = []
for row in df.iterrows():
    row = row[1]
    sample = {
        "text": row['text_EN'],
        "topics": [row['Topic 1'].split("|")[0], row['Topic 2'].split("|")[0]]
    }
    samples.append(sample)





import topic_extraction_methods

from topic_extraction_methods import get_topics, get_topics_spacy, to_BERT_embedding,get_vector_similarity
from fuzzywuzzy import process
import transformers






def validate_topics(document, topics):
    # print(topics, document)
    topic_embeddings = []
    top_score = 0.0
    document_embedding = to_BERT_embedding(document)
    if topics:
        for x in topics:
            topic_embeddings.append(to_BERT_embedding(x))
        scores = []
        for topic_embedding in topic_embeddings:
            similarity = get_vector_similarity(topic_embedding, document_embedding)
            scores.append(similarity)
    #? This top score represents the best similarity between any of the topics and the summary
    return scores
#* FUZZY Search

scores1 = []
scores2 = []
for i, sample in enumerate(samples):
    print(i)
    
    document = sample['text']
    topics = sample['topics']
    score1, score2 = validate_topics(document, topics)
    scores1.append(score1)
    scores2.append(score2)
    # scores.append(score)
    # print(f"Running Average: {np.mean(scores)}  {i+1}/{len(samples)}", end="\r")
print(np.mean(scores1))
print(np.mean(scores2))
#Compare score of extracted topics to Ground truth, which is


#get ratio of keywords to summaries
#compare to average