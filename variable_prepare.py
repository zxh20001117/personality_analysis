import pickle

import pandas as pd

with open("data/value_attributes.pickle", "rb") as f:
    attribute_group = pickle.load(f)


def get_mean_score(scores):
    if len(scores) == 0:
        return None
    else:
        return sum(scores)/len(scores)


def get_abs_diff_score(score):
    if score is None:
        return None
    return abs(score - 3.0)


def get_sentences_count(sentences):
    return len(sentences)


if __name__ == "__main__":
    levels = ["high", "middle"]
    for level in levels:
        data = pd.read_json(f'result/{level} level hotel concat data.json')
        for values in attribute_group.keys():
            data['mean_'+values+'_sentiment'] = data.apply(lambda x: get_mean_score(x[f'{values}_sentiment']), axis=1)
            data['abs_diff_'+values+'_sentiment'] = data.apply(lambda x: get_abs_diff_score(x['mean_'+values+'_sentiment']), axis=1)
            data['sentences_count_'+values] = data.apply(lambda x: get_sentences_count(x[f'{values}_sentences']), axis=1)
        data = data.drop(
            columns=["title", "review", "sentences", "location/transport_sentences", "service_sentences",
                     "food_sentences", "room_sentences", "value_sentences", "facility_sentences",
                     "cleanliness_sentences", "location/transport_sentiment", "service_sentiment", "food_sentiment",
                     "room_sentiment", "value_sentiment", "facility_sentiment", "cleanliness_sentiment"]
            , axis=1
        )
        data.to_json(f'result/{level} level hotel variables.json')
        data.to_excel(f'result/{level} level hotel variables.xlsx')

