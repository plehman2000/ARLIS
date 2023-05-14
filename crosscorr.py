from tqdm import tqdm
import pandas as pd
from topic_extraction_methods import get_topics
import numpy as np
import torch
import spacy


#! Compare results on manual data

df = pd.read_csv("fort.csv")

samples = []

for row in df.iterrows():
    row = row[1]
    sample = {
        "text": row['text_EN'],
        "topics": [row['Topic 1'].split("|")[0], row['Topic 2'].split("|")[0]]
    }
    samples.append(sample)
    
# print(samples[0])


#* extract keywords for each sample



# * Scorers/Metrics
from rouge_score import rouge_scorer
def get_vector_similarity(x1, x2):
    similarity = torch.nn.functional.cosine_similarity(x1=x1, x2=x2, dim=-1)
    return similarity


# rougeL = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

nlp = spacy.load("en_core_web_md")
from topic_extraction_methods import to_longformer_embedding

def calculate_metric(x1,x2, metric="rougeLSum"):
    if metric=="rougeLSum":
        rougeLsum  = rouge_scorer.RougeScorer(['rougeLsum'], use_stemmer=True)
        fmeasure = rougeLsum.score(x1, x2)['rougeLsum'][2]
        return fmeasure
    elif metric == "spacy":
        reference_vector = nlp(x1).vector
        text_vector = nlp(x2).vector
        return get_vector_similarity(torch.tensor(reference_vector), torch.tensor(text_vector))
    elif metric == "longformer":
        reference_vector = to_longformer_embedding(x1)
        text_vector = to_longformer_embedding(x2)
        return get_vector_similarity(torch.tensor(reference_vector), torch.tensor(text_vector))
        
        
    
default_sentence = ""
rouge_L_scores_correlations = []
spacy_wordnet_correlations = []
longformer_embedding_correlations = []

for sample in tqdm(samples):
    topics = get_topics(sample["text"], method="spacy")
    sample['spacy_topics'] = topics
    
    #! Metric calculation
    # ? Rouge-1
    fmeasurex1 = calculate_metric(sample['topics'][0], sample['text'], metric="rougeLSum")
    fmeasurex2 = calculate_metric(sample['topics'][1], sample['text'], metric="rougeLSum")
    fmeasure3 = np.mean([fmeasurex1, fmeasurex2])
    
    fmeasure2 = calculate_metric(" ".join(topics), sample['text'], metric="rougeLSum")
    
    fmeasure1 = calculate_metric(default_sentence, sample['text'], metric="rougeLSum")
    
    rouge_L_scores =[fmeasure3, fmeasure2, fmeasure1]
    rouge_L_scores_r = np.corrcoef(rouge_L_scores, [3,2,1])[0][1]
    
    rouge_L_scores_correlations.append(rouge_L_scores_r)    
    
    
    
    #? Spacy wordnet similarit
    fmeasurex1 = calculate_metric(sample['topics'][0], sample['text'], metric="spacy")
    fmeasurex2 = calculate_metric(sample['topics'][1], sample['text'], metric="spacy")
    fmeasure3 = np.mean([fmeasurex1, fmeasurex2])
    
    fmeasure2 = calculate_metric(" ".join(topics), sample['text'], metric="spacy")
    
    fmeasure1 = calculate_metric(default_sentence, sample['text'], metric="spacy")
    
    spacy_wordnet_scores =[fmeasure3, fmeasure2, fmeasure1]
    spacy_wordnet_r = np.corrcoef(spacy_wordnet_scores, [3,2,1])[0][1]
    
    spacy_wordnet_correlations.append(spacy_wordnet_r) 
    
    #? Longoformer
    fmeasurex1 = calculate_metric(sample['topics'][0], sample['text'], metric="longformer")
    fmeasurex2 = calculate_metric(sample['topics'][1], sample['text'], metric="longformer")
    fmeasure3 = np.mean([fmeasurex1, fmeasurex2])
    
    fmeasure2 = calculate_metric(" ".join(topics), sample['text'], metric="longformer")
    
    fmeasure1 = calculate_metric(default_sentence, sample['text'], metric="longformer")
    
    longformer_scores =[fmeasure3, fmeasure2, fmeasure1]
    longformer_r = np.corrcoef(longformer_scores, [3,2,1])[0][1]
    
    longformer_embedding_correlations.append(longformer_r) 
    # 
    

print(f"rouge_L_scores_correlations: {np.mean(rouge_L_scores_correlations)}")
print(f"spacy_wordnet_correlations: {np.mean(spacy_wordnet_correlations)}")
print(f"longformer_embedding_correlations: {np.mean(longformer_embedding_correlations)}")
#! Calculate relevant metrics


experimental_df = pd.DataFrame(samples)
print(experimental_df.head())

