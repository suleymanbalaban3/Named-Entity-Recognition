import plac
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding


@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)
def main(model=None, output_dir="model", n_iter=100):
    if model is not None:
        nlp = spacy.load(model)  
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  
        print("Created blank 'en' model")


    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)

    else:
        ner = nlp.get_pipe("ner")

    TRAIN_DATA = preprocessData("data/engtrain.bio")
    TEST_DATA = preprocessData("data/engtest.bio")

    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])


    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    with nlp.disable_pipes(*other_pipes):  

        if model is None:
            nlp.begin_training()
        i = 0
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  
                    annotations,  
                    drop=0.5,  
                    losses=losses,
                )
            print("Iter ", str(i+1), " Losses " ,  losses)
            i += 1

    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

    print("----------------------- Testing Results -----------------------")
    print("Loaded model from ", output_dir)
    loaded_ner_model = spacy.load(output_dir)
    print(evaluate(loaded_ner_model, TEST_DATA))

def evaluate(model, data):
    scorer = Scorer()
    for input_, annot in data:
        doc_gold_text = model.make_doc(input_)
        gold = GoldParse(doc_gold_text, entities=annot['entities'])
        pred_value = ner_model(input_)
        scorer.score(pred_value, gold)
    return scorer.scores    

def preprocessData(input_path):
    wordIndex = 0
    entities = []
    sentenceEntityDic = {}
    sentence = ""
    sentenceFeatures = []
    training_data = []

    try:
        f=open(input_path,'r') 
        def prepareSentenceFeatures(sentence, entities, training_data):
            sentence = sentence.replace("\n", " ")
            training_data.append(tuple([sentence, {'entities' : entities} ]))

        for line in f:
            if(line == '\n'):
                prepareSentenceFeatures(sentence, entities, training_data)

                entities = []
                sentence = ""
                wordIndex = 0
            else:
                tokens = line.split("\t")
                sentence += tokens[1] 
                entity = [wordIndex, wordIndex + len(tokens[1])-1, tokens[0]]
                wordIndex += len(tokens[1])
                entities.append(tuple(entity))

        return training_data        
    except Exception as e:
        print("Exception catched while reading data and preprocessing!")
        return None

if __name__ == "__main__":
    plac.call(main)