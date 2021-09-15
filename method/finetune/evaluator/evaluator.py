# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
import logging
import sys
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score

def read_answers(filename):
    answers={}
    with open(filename) as f:
        for line in f:
            line=line.strip()
            idx1,idx2,label=line.split()
            answers[(idx1,idx2)]=label
    return answers

def read_predictions(filename):
    predictions={}
    with open(filename) as f:
        for line in f:
            line=line.strip()
            idx1,idx2,label=line.split()
            predictions[(idx1,idx2)]=label
    return predictions

def calculate_scores(answers,predictions):
    y_trues,y_preds=[],[]
    for key in answers:
        if key not in predictions:
            logging.error("Missing prediction for ({},{}) pair.".format(key[0],key[1]))
            sys.exit()
        y_trues.append(answers[key])
        y_preds.append(predictions[key])
    scores={}
    scores['acc'] = accuracy_score(y_trues, y_preds)
    scores['Prediction']=precision_score(y_trues, y_preds, average='macro')
    scores['Recall']=recall_score(y_trues, y_preds, average='macro')
    scores['F1']=f1_score(y_trues, y_preds, average='macro')
    # scores['Recall'] = recall_score(y_trues, y_preds, pos_label='1')
    # scores['Prediction'] = precision_score(y_trues, y_preds, pos_label='1')
    # scores['F1'] = f1_score(y_trues, y_preds, pos_label='1')
    return scores

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate leaderboard predictions for BigCloneBench dataset.')
    parser.add_argument('--answers', '-a',help="filename of the labels, in txt format.")
    parser.add_argument('--predictions', '-p',help="filename of the leaderboard predictions, in txt format.")
    parser.add_argument("--output_path", '-o', default=None, type=str, help="Result path.")

    args = parser.parse_args()
    answers=read_answers(args.answers)
    predictions=read_predictions(args.predictions)
    scores=calculate_scores(answers,predictions)
    print(scores)
    with open(args.output_path, 'w') as f:
        f.write(str(scores))

if __name__ == '__main__':
    main()
