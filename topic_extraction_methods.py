
# from bertopic import BERTopic
# from sklearn.datasets import fetch_20newsgroups
import spacy
from transformers import (
    TokenClassificationPipeline,
    AutoModelForTokenClassification,
    AutoTokenizer,
)
from transformers.pipelines import AggregationStrategy
import numpy as np

# Define keyphrase extraction pipeline
class KeyphraseExtractionPipeline(TokenClassificationPipeline):
    def __init__(self, model, *args, **kwargs):
        super().__init__(
            model=AutoModelForTokenClassification.from_pretrained(model),
            tokenizer=AutoTokenizer.from_pretrained(model),
            *args,
            **kwargs
        )

    def postprocess(self, model_outputs):
        results = super().postprocess(
            model_outputs,
            aggregation_strategy=AggregationStrategy.SIMPLE,
        )
        return np.unique([result.get("word").strip() for result in results])

# Load pipeline
model_name = "ml6team/keyphrase-extraction-kbir-inspec"
extractor = KeyphraseExtractionPipeline(model=model_name)


def get_topics(text):
    # ? using miniLM
    topics = extractor(text)
    topics = list(topics)
    topics = [x.lower() for x in topics]
    return topics




# Perform standard imports import spacy nlp = spacy.load('en_core_web_sm')
# Write a function to display basic entity info: def show_ents(doc): if doc.ents: for ent in doc.ents: print(ent.text+' - ' +str(ent.start_char) +' - '+ str(ent.end_char) +' - '+ent.label_+ ' - '+str(spacy.explain(ent.label_))) else: print('No named entities found.')
NER = spacy.load("en_core_web_sm")



def get_topics_spacy(text):
    doc = NER(text)
    topic_entities = []
    for x in doc.ents:
        if x.label_ not in ["MONEY", "DATE"]:
            topic_entities.append(x)

    return topic_entities


text = """Dominion Voting Systems agrees to a $787 million settlement
 in their lawsuit against Fox News over defamation claims from the 2020 United States presidential election."""

print(get_topics_spacy(text))