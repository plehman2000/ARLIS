
from fuzzywuzzy import process
from datasets import load_dataset
from tqdm import tqdm
from bertopic import BERTopic
import spacy


dataset = load_dataset("xsum")

train = dataset['train']

print(type(train))
samples = []
for i, sample in enumerate(tqdm(train)):
    samples.append(sample['document'])
    if i == len(train)//10:
        break
        # {
        # "document": sample['document'],
        #  "summary": sample['summary']
        #  })



nlp = spacy.load("en_core_web_md", exclude=['tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer'])

topic_model = BERTopic(embedding_model=nlp, verbose=True)
topics, probs = topic_model.fit_transform(samples)

print(topic_model.get_topic_info())

print(topic_model.get_document_info(samples))