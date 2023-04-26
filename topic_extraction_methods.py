# from bertopic import BERTopic
# from sklearn.datasets import fetch_20newsgroups
import spacy
import torch
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
            model_outputs, aggregation_strategy=AggregationStrategy.SIMPLE,
        )
        return np.unique([result.get("word").strip() for result in results])


# Load pipeline
model_name = "ml6team/keyphrase-extraction-kbir-inspec"
extractor = KeyphraseExtractionPipeline(model=model_name)


def get_topics(text):
    # ? using miniLM
    topics = extractor(text)
    topics = list(topics)
    topics = [str(x.lower()) for x in topics]
    return topics


from transformers import BertTokenizer, BertModel, LongformerTokenizer, LongformerModel

tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
# Load pre-trained model (weights)
model = LongformerModel.from_pretrained(
    "allenai/longformer-base-4096",
    output_hidden_states=True,  # Whether the model returns all hidden-states.
)

# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()


def to_BERT_embedding(text):

    # Add the special tokens.
    marked_text = "[CLS] " + text + " [SEP]"

    # Split the sentence into tokens.
    tokenized_text = tokenizer.tokenize(marked_text)
    # print(len(tokenized_text))
    if len(tokenized_text) > 4096:
        tokenized_text = tokenized_text[:4096]
    # Map the token strings to their vocabulary indeces.
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_ids = [1] * len(tokenized_text)
    segments_tensors = torch.tensor([segments_ids])
    with torch.no_grad():

        outputs = model(tokens_tensor, segments_tensors)

        # Evaluating the model will return a different number of objects based on
        # how it's  configured in the `from_pretrained` call earlier. In this case,
        # becase we set `output_hidden_states = True`, the third item will be the
        # hidden states from all layers. See the documentation for more details:
        # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
        hidden_states = outputs[2]
        hidden_states = torch.stack(hidden_states, dim=0)

    token_vecs = hidden_states[-2][0]
    # Calculate the average of all 22 token vectors.
    embedding = torch.mean(token_vecs, dim=0)
    return embedding


# Perform standard imports import spacy nlp = spacy.load('en_core_web_sm')
# Write a function to display basic entity info: def show_ents(doc): if doc.ents: for ent in doc.ents: print(ent.text+' - ' +str(ent.start_char) +' - '+ str(ent.end_char) +' - '+ent.label_+ ' - '+str(spacy.explain(ent.label_))) else: print('No named entities found.')
NER = spacy.load("en_core_web_sm")


def get_topics_spacy(text):
    doc = NER(text)
    topic_entities = []
    for x in doc.ents:
        if x.label_ not in ["MONEY", "DATE"]:
            topic_entities.append(str(x.text))

    return topic_entities


def get_vector_similarity(x1, x2):
    similarity = torch.nn.functional.cosine_similarity(x1=x1, x2=x2, dim=-1)
    return similarity


# text1 = """Dominion Voting Systems agrees to a $787 million settlement
#  in their lawsuit against Fox News over defamation claims from the 2020 United States presidential election."""

# text2 = """
# Fox News reaches last-minute settlement with Dominion
# """
# text1_embedding = to_BERT_embedding(text1)
# text2_embedding = to_BERT_embedding(text2)
# # print(text1_embedding.shape)
# print(get_vector_similarity(x1=text1_embedding, x2=text2_embedding))

# docid = 10
# latin_sq = 17
# example = """
# "The narrative about LGBT people in Poland sounds like a disappointing Qatarin, who bases all its power on an exaggerated and in fact that harm to" celebrating suffering. " That is why it was necessary to finally say "enough". As I hate naming women-members' names of their colleagues from the industry, but in the skirt, I avoid calling a rainbow night, i.e. a protest against the arrest of Margot and his brutal suppression by the police, Polish Stonewall. Although these key events for the LGBT community have a lot of in common, they are different in two basic issues: time and local context, which the Vistula homo-, bi- and transfobia make an extremely unique discrimination at the international arena. Well, unless we compare it with the Putin regime , but not for such standards should be pursued by European and apparently still members a state of the Union. Successism of impotents, i.e. the case of Margot and the declining sanacja also the impotents of the impotents, i.e. the case of Margot and the decline, so I will dwell on whether systemic discrimination and violence against LGBT people in Poland is original or twin -like, or a twin -like one to Russian, nor over the similarities between the New York and Warsaw uprising in defense of the rainbow flag, which is trying to stain the grenade of a servile uniform and brown, and-worse-white and red colors appropriated by hatred sowers. I think that what happened on August 7, 2020 (and therefore over Half a century after demonstrations in the USA) in the heart of Warsaw is on the one hand a great outflow of social, devoid of empathy conscience, and on the other - for the LGBT community and its allies - a specific, strong symbol that does not need any Western labels to have strength to have strength destroying and stop our attention to the discussion of something as fundamental as the rights man Ka. Apart from how much a dramatic and traumatic event we were dealing with then, the relationships of its participants and participants, which Amnesty International collected in the recently published and described report, we were treated as criminals. From the atmosphere of hostility to the harassment of people defending LGBTI rights, but also in the moving and produced by the campaign against the homophobia film by Michał Bolland. Citivist in Poland: Criminal and future emigrant also an emigrant in Poland: a criminal and future emigrantpauline Januszewskadokument 7 August on the second anniversary of the discussed here The manifestation and everything that followed her, from the festival travels, finally came to the universal circulation. You can, in full and completely free, and even need to be seen on the web, but not because it is an outstanding cinema. An unlucky half -hour projection is, regardless of their artistic qualities, an important testimony of people who fought for dignity two years ago their and Margot, and then they were punished: beaten, detained in custody, drawn in the courts - as if everyday life in ultra -conservative and intolerant Poland was not enough for them. "Good, Januszewska, finish these sadness, how much you can go around Repeat the same, "you will think. And rightly so, because without watching Bolland's movie, it is known how badly to people standing out from the standards of ruling and determining the social situation of ultra -conservatives are leading in a country that has long been in the tail of countries interested in supporting equality. But I mention it for a slightly different reason - To show that the narrative about LGBT people in Poland sounds like a disappointing Qatarin, who bases all its power on an exaggerated and in fact harmful - as I once read in a biweekly - "celebrating suffering". "Some say that suffering ennobles, but I think that suffering is primarily paralyzed," wrote Bartosz Żurawiecki, leaning over the queer threads of Polish cinema. And I will say that I not only agree with him, but I also consider this martyrdom rhetoric as dominating both in the texts of culture and the public debate. People, do not ideologize the Żurawieckiejamiekzamami, testing the theses of the said author, I must notice that in these stories good and deserves Acceptance and support of: gay, lesbian, non -liner, bisexual, transgender or asexual person should be quiet, modest, unknown to the concept of exaggeration and (God, defend!) This terrible and abused in the rainbow context of inflection. So they must act as victims, which can simply become dead at any time from stuck and discriminated. If - forgive me the use of a right -wing nomenclature - they wore colorful feathers in the ass, they themselves released. However, if they "did not flaunt" with their presence, then we can sympathize with them or pretend that there are simply non -heterosexual people. Most minority and systemically oppressed groups, this pattern is still repeating. For example, when we discuss loudly about reproductive rights, we usually absolve abortion, if a potential mother suffers enough. When the pregnancy is removed by the singer with too small to the room of the next child with an apartment or a girl who decides to surgery with any (not your interest), but not related to any drama, you need to start a public lick. Don't get me wrong. I do not underestimate the fact that due to the lack of access to legal abortion and fasting for LGBT and refusing their basic rights, people die. Because-as the heroine of the film of the 7th August says-in a dehumanizing and shot by alleged, self-proclaimed defenders of life, the story of "Rainbow plague and ideology" we all believed. As a result, on the rainbow side "they kill" (a report of the campaign against homophobia indicates that as many as 55 percent of this community has suicidal thoughts), and the rest "goes to kill". Homophobia kills the ham also quietly They need nobody's pity, they do not have to earn your attention, help, tolerance and false compromise life permits in society. That is why it was necessary to say "enough" to suffering. Daming a good matter and non -tracking marches of equality of gay colleagues? Because they threw out the vulnerable sense of dignity and subjectivity, relief and humility in the basket. Rightly? I will answer with a quote from John Paul II: "I still like!" Anger, and in all the media there is a thread of a person who unceremoniously storms binary order, does not play in courtesy conventions, does not pick up words and states: "say that I fucking Poland is not to say anything." Pride, radicalism and pissed off are completely on the spot, because if because of who you love, you can hear on the street (straight from the Homofobus speaker) that you rape small children, threaten the Polish family and Poland in general, then you have absolute right to shout and demand Justice even with rainbow feathers in the butt, and not listen to the fairy tales of politicians from the opposition who still say that our society is not ready for something. On August 7, 2020, did this change? Has it woke up the need to protest harder? In my opinion - yes, which is also indicated by the fact that the power chasing activists for rainbow flags and Jarosław Kaczyński mocking from transgender in their pre -election rallies.
#  However, I wonder if Poles and Poles are still buying it. Definitely confusion. Anthology of Polish literature Queeroprac.
#   The community, unfortunately, in the film by Michał Bolland, who - as I read in the description of the production - asks
#    a similar question, did the rainbow night "really awakened Queer Rage in Poland?" - I don't find answers. And I do not
#     blame the heroes and heroines, because Kajetan, Kamila, Cream and Julia and her partner Kalina show great courage, talking
#      about their experiences from two years ago, as well as the everyday life of being a non -heteronormative person in a
#       heteronormative society. Ba, show that participation in the film has a therapeutic and allowing to overcome the trauma
#        and stagnation power, which testifies to their great strength, as well as the desire to continue to act for their rights.
#         However, this ends with the director's interest in the influence of the August events on reality, what It only shows that
#          it is still difficult to overcome the messianic cultural traditions on the Vistula. And subsequent formats of stories reaching
#           for queer themes, such as the Netflix Queen, also do not pass the test for such a needed and power -free uncompromisingism, and
#            instead choose paralyzing patients. "Queen" from Netflix turned out to be a false alliance Okuńskapo ally of sessions similar
#             to the mentioned here, I feel like crying and hugging everyone who destroys hatred. Meanwhile, tears are not enough to fight
#              the latter. You need pride, strength and radicalism, thanks to which there will be no more victims and corpses. One of the
#               heroines of the Bolland movie says: "I'm not saying that everyone is to love us, but I would like them not to hate us." And
#                I would add: "And that they would stop feeling sorry for us and began to see how equals themselves." The tolerance
#                 with equality flowing from pity has nothing to do. "
#     """

# topics = ["LGBT Discrimination", "LGBT protests"]
# example_embedding = to_BERT_embedding(example)



# for x in topics:
#     topic_embedding = to_BERT_embedding(x)
#     similarity = get_vector_similarity(example_embedding, topic_embedding)
#     print(similarity, x)
