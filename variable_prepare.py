import math
import pickle

import pandas as pd

with open("data/value_attributes.pickle", "rb") as f:
    attribute_group = pickle.load(f)

personalities = ['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']


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


def get_personality_level(data):
    personality_avg = {}
    for personality in personalities:
        personality_avg[personality] = data[personality].mean()
    for personality in personalities:
        data[f'{personality}_level'] = data[personality].apply(lambda x: 1 if x > personality_avg[personality] else 0)


def get_if_mind_values(data):
    for values in attribute_group.keys():
        data[f'if_{values}_mind'] = data.apply(lambda x: 1 if x['sentences_count_'+values] > 0 else 0, axis=1)


def get_abs_diff_score_by_hotel(data):
    group_data = data.groupby('hotel_name')
    hotel_mean = {}
    for hotel in group_data.groups.keys():
        temp_dict = {}
        for values in attribute_group.keys():
            value_list = []
            for line in group_data.get_group(hotel)[f'{values}_sentiment']:
                for i in line:
                    value_list.append(i)
            temp_dict[values] = sum(value_list)/len(value_list)
        hotel_mean[hotel] = temp_dict

    for values in attribute_group.keys():
        abs_hotel_diff_score_list = []
        mean_hotel_diff_score_list = []
        for i in range(len(data)):
            score = data.iloc[i]['mean_' + values + '_sentiment']
            mean_score = hotel_mean[data.iloc[i]['hotel_name']][values]
            if score is None:
                abs_hotel_diff_score_list.append(None)
                mean_hotel_diff_score_list.append(None)
            else:
                abs_hotel_diff_score_list.append(
                    abs(score - mean_score)
                )
                mean_hotel_diff_score_list.append(score - mean_score)
        data[f'abs_hotel_{values}_sentiment'] = abs_hotel_diff_score_list
        data[f'mean_hotel_{values}_sentiment'] = mean_hotel_diff_score_list


def get_ln(score):
    if score is None:
        return None
    else:
        return math.log(score)


if __name__ == "__main__":
    hotel_data = pd.read_excel("data/hotel level merged.xlsx")
    hotel_links_data = pd.read_excel("data/hotel level links.xlsx")
    levels = ["high", "middle"]
    data_list = []
    for level in levels:
        data = pd.read_json(f'result/{level} level hotel concat data.json')
        data = data.dropna(axis=0, subset=personalities)
        for values in attribute_group.keys():
            data['mean_'+values+'_sentiment'] = data.apply(lambda x: get_mean_score(x[f'{values}_sentiment']), axis=1)
            data['abs_diff_'+values+'_sentiment'] = data.apply(lambda x: get_abs_diff_score(x['mean_'+values+'_sentiment']), axis=1)
            data['sentences_count_'+values] = data.apply(lambda x: get_sentences_count(x[f'{values}_sentences']), axis=1)
            data['ln_mean_'+values+'_sentiment'] = data.apply(lambda x: get_ln(x['mean_'+values+'_sentiment']), axis=1)
        get_personality_level(data)
        get_if_mind_values(data)

        data = pd.merge(data, hotel_data[['id_review', 'hotel_name']], on='id_review', how='left').copy()

        get_abs_diff_score_by_hotel(data)

        data = pd.merge(data, hotel_links_data, on='hotel_name', how='left').copy()

        data = data.drop(
            columns=["title", "review", "sentences", "location/transport_sentences", "service_sentences",
                     "food_sentences", "room_sentences", "value_sentences", "facility_sentences",
                     "cleanliness_sentences", "location/transport_sentiment", "service_sentiment", "food_sentiment",
                     "room_sentiment", "value_sentiment", "facility_sentiment", "cleanliness_sentiment"]
            , axis=1
        )
        
        data.to_json(f'result/{level} level hotel variables.json')
        data.to_excel(f'result/{level} level hotel variables.xlsx')
        data_list.append(data)

    all_data = pd.concat(data_list)
    all_data.reset_index(inplace=True, drop=True)
    all_data.to_json('result/all level hotel variables.json')
    all_data.to_excel('result/all level hotel variables.xlsx')
    all_data.to_csv('result/all level hotel variables.csv') 

