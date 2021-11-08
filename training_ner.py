import nltk
import spacy
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import conlltags2tree,tree2conlltags
from pprint import pprint
import en_core_web_sm
nlp = en_core_web_sm.load()
# print(nlp.pipe_names)

ner = nlp.get_pipe('ner')

filename = 'C:/Users/adars/Desktop/TOC_Assignment_Instructions_Code/dataset'
file = open(filename, 'rb')
TRAIN_DATA= file.read()
file.close()

for _, annotations in TRAIN_DATA:
  for ent in annotations.get("entities"):
    ner.add_label(ent[2])

# Disable pipeline components you dont need to change
pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

# Import requirements
import random
from spacy.util import minibatch, compounding
from pathlib import Path
from spacy.training.example import Example

# TRAINING THE MODEL
with nlp.disable_pipes(*unaffected_pipes):

  # Training for 30 iterations
  for iteration in range(40):

    # shuufling examples  before every iteration
    random.shuffle(TRAIN_DATA)
    losses = {}
    
    """
    # batch up the examples using spaCy's minibatch
    batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
    for batch in batches:
        texts, annotations = zip(*batch)
        nlp.update(texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    losses=losses)
    """
        
        
    for batch in spacy.util.minibatch(TRAIN_DATA, size=2):
        for text, annotations in batch:
            # create Example
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            # Update the model
            nlp.update([example], losses=losses, drop=0.3)
        
        
        print("Losses", losses)
        
# Save the  model to directory
output_dir = Path('C:/Users/adars/Desktop/TOC_Assignment_Instructions_Code/')
nlp.to_disk(output_dir)
print("Saved model to", output_dir)

# Load the saved model and predict
print("Loading from", output_dir)
nlp_updated = spacy.load(output_dir)
doc = nlp_updated("Raghav has to give access of his windows account to Shrishail to execute an application")
print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
