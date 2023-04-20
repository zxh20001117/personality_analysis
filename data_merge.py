import pickle

import pandas as pd

def concat_level_hotel_data(level):
    detail_data = pd.read_json(f"data/{level} level hotel data.json")
    iovo_data = pd.read_json(f"data/{level} hotel iovo scores.json")
    personality_data = pd.read_csv(f"data/{level} level hotel personality predict.csv")

    detail_data.reset_index(drop=True, inplace=True)
    personality_data.reset_index(drop=True, inplace=True)

    print(detail_data.columns)
    print(personality_data.columns)
    print(iovo_data.columns)

    with open("data/value_attributes.pickle", "rb") as f:
        attribute_group = pickle.load(f)

    sentiment_results = {}
    for i in attribute_group.keys():
        sentiment_results[i] = []

    sentence_count = 0

    for values in attribute_group.keys():
        for line in detail_data[f'{values}_sentences']:
            temp_result = []
            for sentence in line:
                temp_result.append(iovo_data.iloc[sentence_count]['IOVO predict'])
                sentence_count += 1
            sentiment_results[values].append(temp_result)

    print(len(detail_data))
    for key in sentiment_results.keys():
        print(len(sentiment_results[key]))
        personality_data[f'{key}_sentiment'] = sentiment_results[key]

    concat_data = pd.concat([detail_data, personality_data], axis=1)
    concat_data.to_json(f'result/{level} level hotel concat data.json')
    concat_data.to_excel(f'result/{level} level hotel concat data.xlsx')
    print(len(concat_data), len(detail_data), len(personality_data))

# sentiment_results = pd.DataFrame(sentiment_results)


if __name__ == "__main__":
    level = ["high", "middle"]
    for i in level:
        concat_level_hotel_data(i)
