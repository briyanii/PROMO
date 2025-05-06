import os
import re
import pickle
import argparse

import multiprocessing as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

def decrement_id(x):
    return x - 1

def load_kuairand():
    _dir = 'KuaiRand'
    interaction_features = 'log_standard_4_08_to_4_21_pure.csv'
    user_features = "user_features_pure.csv"
    item_features = 'video_features_basic_pure.csv'
    item_statistic_features = "video_features_statistic_pure.csv"
    user_features = os.path.join(_dir, user_features)
    user_features = pd.read_csv(user_features)
    item_features = os.path.join(_dir, item_features)
    item_features = pd.read_csv(item_features)
    item_statistic_features = os.path.join(_dir, item_statistic_features)
    item_statistic_features = pd.read_csv(item_statistic_features)
    interaction_features = os.path.join(_dir, interaction_features)
    interaction_features = pd.read_csv(interaction_features)

    return user_features, item_features, item_statistic_features, interaction_features

def load_amazon(subset):
    review = load_dataset('McAuley-Lab/Amazon-Reviews-2023', f'raw_review_{subset}', trust_remote_code=True, split='full')
    meta = load_dataset('McAuley-Lab/Amazon-Reviews-2023', f'raw_meta_{subset}', trust_remote_code=True, split='full')

    item_ids = []
    prices = []
    has_prices = []
    rating_numbers = []
    average_ratings = []
    for e in tqdm(meta):
        item_ids.append(e['parent_asin'])

        price = e['price']
        price_candidates = re.findall('\d+\.*\d*', price)
        has_price = 1 if len(price_candidates) > 0 else 0
        price = float(price_candidates[0]) if has_price else 0.0

        prices.append(price)
        has_prices.append(has_price)
        rating_numbers.append(e['rating_number'])
        average_ratings.append(e['average_rating'])

    item_features = pd.DataFrame({
        'item_id': item_ids,
        'price': prices,
        'has_price': has_prices,
        'rating_number': rating_numbers,
        'average_rating': average_ratings,
    })

    user_ids = []
    item_ids = []
    timestamps = []
    ratings = []

    #i = 0
    for e in tqdm(review):
        #if i == 10000: break
        #i += 1
        user_ids.append(e['user_id'])
        item_ids.append(e['parent_asin'])
        timestamps.append(e['timestamp'])
        ratings.append(e['rating'])

    interaction_features = pd.DataFrame({
        'item_id': item_ids,
        'user_id': user_ids,
        'timestamp': timestamps,
        'rating': ratings,
    })

    user_ids = list(set(user_ids))
    user_features = pd.DataFrame({
        'user_id': user_ids,
    })

    remap_user_id = {v:k for k,v in user_features['user_id'].to_dict().items()}
    remap_item_id = {v:k for k,v in item_features['item_id'].to_dict().items()}

    item_features['item_id'] = item_features['item_id'].apply(lambda x: remap_item_id[x])
    user_features['user_id'] = user_features['user_id'].apply(lambda x: remap_user_id[x])
    interaction_features['item_id'] = interaction_features['item_id'].apply(lambda x: remap_item_id[x])
    interaction_features['user_id'] = interaction_features['user_id'].apply(lambda x: remap_user_id[x])

    return user_features, item_features, interaction_features


def load_movielens100k():
    _dir = 'MovieLens100k'
    user_features = 'u.user'
    item_features = 'u.item'
    interaction_features = 'u.data'
    occupations = 'u.occupation'

    occupations = os.path.join(_dir, occupations)
    with open(occupations, 'r') as fp:
        occupations = [x.strip() for x in fp.readlines()]

    interaction_features = os.path.join(_dir, interaction_features)
    interaction_features = pd.read_csv(interaction_features, sep='\t', header=None)
    interaction_features.columns = ['user_id', 'item_id', 'rating', 'timestamp']
    interaction_features['item_id'] = interaction_features['item_id'].apply(decrement_id)
    interaction_features['user_id'] = interaction_features['user_id'].apply(decrement_id)

    user_features = os.path.join(_dir, user_features)
    user_features = pd.read_csv(user_features, sep='|', header=None)
    user_features.columns = [
        'user_id',
        'age',
        'gender', #string
        'occupation', #string
        'zip_code', #string
    ]
    user_features['user_id'] = user_features['user_id'].apply(decrement_id)
    user_features['occupation'] = user_features['occupation'].apply(
        lambda x: occupations.index(x)
    )

    item_features = os.path.join(_dir, item_features)
    encoding = 'iso-8859-1'
    item_features = pd.read_csv(item_features, sep='|', encoding=encoding, header=None)
    item_features.columns = [
        'movie_id',
        'movie_title', #string
        'release_date', #string, hasnan
        'video_release_date', # all nan ?
        'IMDb_URL', # string, hasnan
        'unknown',
        'Action',
        'Adventure',
        'Animation',
        "Children's",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film-Noir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western",
    ]
    item_features['movie_id'] = item_features['movie_id'].apply(decrement_id)

    return user_features, item_features, interaction_features

def load_movielens1m():
    _dir = 'MovieLens1m'
    user_features = os.path.join(_dir, 'users.dat')
    user_features = pd.read_csv(user_features, sep='::', header=None)
    user_features.columns = [
        'user_id',
        'gender',
        'age',
        'occupation',
        'zipcode',
    ]

    item_features = os.path.join(_dir, 'movies.dat')
    encoding = 'iso-8859-1'
    item_features = pd.read_csv(item_features, sep='::', header=None, encoding=encoding)
    item_features.columns = [
        'movie_id',
        'title',
        'genres',
    ]


    genres = [
        "Action",
        "Adventure",
        "Animation",
        "Children's",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film-Noir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western",
    ]

    def process_genre(x):
        ft = [0] * (len(genres) + 1)
        for g in x.split('|'):
            idx = genres.index(g) + 1
            ft[idx] = 1
        return ft

    genre_multihot = item_features['genres'].apply(process_genre)
    genre_multihot = np.vstack(genre_multihot.values)
    genres = ['unknown'] + genres
    item_features[genres] = genre_multihot
    item_features[genres]

    interaction_features = os.path.join(_dir, 'ratings.dat')
    interaction_features = pd.read_csv(interaction_features, sep='::', header=None)
    interaction_features.columns = [
        'user_id',
        'item_id',
        'rating',
        'timestamp'
    ]

    remap_user_id = {v:k for k,v in user_features['user_id'].to_dict().items()}
    remap_item_id = {v:k for k,v in item_features['movie_id'].to_dict().items()}

    user_features['user_id'] = user_features['user_id'].apply(lambda x: remap_user_id[x])
    item_features['movie_id'] = item_features['movie_id'].apply(lambda x: remap_item_id[x])
    interaction_features['user_id'] = interaction_features['user_id'].apply(lambda x: remap_user_id[x])
    interaction_features['item_id'] = interaction_features['item_id'].apply(lambda x: remap_item_id[x])

    return user_features, item_features, interaction_features

def load_data(ds_name):
    if ds_name == 'KuaiRand':
        return load_kuairand()
    elif ds_name == 'MovieLens100k':
        return load_movielens100k()
    elif ds_name == 'MovieLens1m':
        return load_movielens1m()
    elif ds_name == 'AmazonGames':
        return load_amazon('Video_Games')
    elif ds_name == 'AmazonFashion':
        return load_amazon('Amazon_Fashion')
    else:
        raise Exception("Not Implemented")

def preprocess_kuairand_user_features(user_features):
    feat_meta = [
        {'name': 'user_id', 'type': 'id', 'count': user_features.shape[0]},
        # binary
        {'name': 'is_live_streamer', 'type': 'categoric', 'count': 2, 'values': [0, 1], 'default': 0},
        {'name': 'is_video_author', 'type': 'categoric', 'count': 2},
        {'name': 'onehot_feat0', 'type': 'categoric', 'count': 2, 'default': 0},
        {'name': 'onehot_feat12', 'type': 'categoric', 'count': 2, 'default': 0},
        {'name': 'onehot_feat13', 'type': 'categoric', 'count': 2, 'default': 0},
        {'name': 'onehot_feat14', 'type': 'categoric', 'count': 2, 'default': 0},
        {'name': 'onehot_feat15', 'type': 'categoric', 'count': 2, 'default': 0},
        {'name': 'onehot_feat16', 'type': 'categoric', 'count': 2, 'default': 0},
        {'name': 'onehot_feat17', 'type': 'categoric', 'count': 2, 'default': 0},
        # one_hot
        {'name': 'user_active_degree', 'type': 'categoric', 'count': 4, 'values': ['full_active', 'high_active', 'middle_active', 'low_active'], 'default': 'low_active'},
        {'name': 'onehot_feat6', 'type': 'categoric', 'count': 3},
        # to embeddings
        {'name': 'follow_user_num_range', 'type': 'categoric', 'count': 8, 'values': ['0', '(0,10]', '(10,50]', '(50,100]','(100,150]', '(150,250]', '(250,500]', '500+']},
        {'name': 'fans_user_num_range', 'type': 'categoric', 'count': 7, 'values': ['0', '[1,10)', '[10,100)', '[100,1k)', '[1k,5k)', '[5k,1w)', '1w+'], 'default': '1w+'},
        {'name': 'friend_user_num_range', 'type': 'categoric', 'count': 7, 'values': ['0', '[1,5)', '[5,30)', '[30,60)', '[60,120)', '[120,250)', '250+']},
        {'name': 'register_days_range', 'type': 'categoric', 'count': 7, 'values': ['-30', '31-60', '61-90', '91-180', '181-365', '366-730', '730+'], 'default': '-30'},
        {'name': 'onehot_feat1', 'type': 'categoric', 'count': 7, 'default': 0},
        {'name': 'onehot_feat2', 'type': 'categoric', 'count': 50, 'default': 0},
        {'name': 'onehot_feat3', 'type': 'categoric', 'count': 1471, 'default': 0},
        {'name': 'onehot_feat4', 'type': 'categoric', 'count': 15, 'default': 0},
        {'name': 'onehot_feat5', 'type': 'categoric', 'count': 34, 'default': 0},
        {'name': 'onehot_feat7', 'type': 'categoric', 'count': 118, 'default': 0},
        {'name': 'onehot_feat8', 'type': 'categoric', 'count': 454, 'default': 0},
        {'name': 'onehot_feat9', 'type': 'categoric', 'count': 7, 'default': 0},
        {'name': 'onehot_feat10', 'type': 'categoric', 'count': 5, 'default': 0},
        {'name': 'onehot_feat11', 'type': 'categoric', 'count': 5, 'default': 0},
    ]
    columns = []
    for entry in feat_meta:
        ft_name = entry['name']
        columns.append(ft_name)
        if entry['type'] == 'categoric' and 'default' in entry and 'values' in entry:
            user_features[ft_name] = user_features[ft_name].apply(
                lambda x: x if x in entry['values'] else entry['default']
            )
        elif entry['type'] == 'categoric' and 'default' in entry:
            user_features[ft_name] = user_features[ft_name].apply(
                lambda x: int(x) if x >= 0 else entry['default']
            )
        if entry['type'] == 'categoric' and 'values' in entry:
            user_features[ft_name] = user_features[ft_name].apply(
                lambda x: entry['values'].index(x)
            )

    final = user_features[columns]
    final = final.drop(columns=['user_id'])

    meta = pd.DataFrame(feat_meta)[['name', 'type', 'count']]
    return final, meta

def preprocess_kuairand_item_features(item_features, item_statistic_features):
    def video_duration_process(x):
        if x < 10000:
            return 0
        elif x < 50000:
            return 1
        elif x < 100000:
            return 2
        else:
            return 3
    def tag_process(x):
        if isinstance(x, str):
            return int(x.split(',')[0])
        return int(x) if x >= 0 else 0

    feat_meta_basic = [
        {'name': 'video_id', 'type': 'id', 'count': item_features.shape[0]},
        {'name': 'video_type', 'type': 'categoric', 'count': 2, 'values': ['NORMAL', 'AD'], 'default': 'NORMAL'},
        {'name': 'video_duration', 'type': 'categoric', 'count': 4, 'preprocess': video_duration_process},
        {'name': 'music_type', 'type': 'categoric', 'count': 6, 'values': [9.0, 4.0, 8.0, 7.0, 11.0, 'OTHER'], 'default': 'OTHER'},
        {'name': 'tag', 'type': 'categoric', 'count': 69, 'preprocess': tag_process},
    ]
    feat_meta_stat = [
        {'name': 'video_id', 'type': 'id', 'count': item_statistic_features.shape[0]},
        {'name': 'counts', 'type': 'numeric'},
        {'name': 'show_cnt', 'type': 'numeric'},
        {'name': 'show_user_num', 'type': 'numeric'},
        {'name': 'play_cnt', 'type': 'numeric'},
        {'name': 'play_user_num', 'type': 'numeric'},
        {'name': 'play_duration', 'type': 'numeric'},
        {'name': 'complete_play_cnt', 'type': 'numeric'},
        {'name': 'complete_play_user_num', 'type': 'numeric'},
        {'name': 'valid_play_cnt', 'type': 'numeric'},
        {'name': 'valid_play_user_num', 'type': 'numeric'},
        {'name': 'long_time_play_cnt', 'type': 'numeric'},
        {'name': 'long_time_play_user_num', 'type': 'numeric'},
        {'name': 'short_time_play_cnt', 'type': 'numeric'},
        {'name': 'short_time_play_user_num', 'type': 'numeric'},
        {'name': 'play_progress', 'type': 'numeric'},
        {'name': 'comment_stay_duration', 'type': 'numeric'},
        {'name': 'like_cnt', 'type': 'numeric'},
        {'name': 'like_user_num', 'type': 'numeric'},
        {'name': 'click_like_cnt', 'type': 'numeric'},
        {'name': 'double_click_cnt', 'type': 'numeric'},
        {'name': 'cancel_like_cnt', 'type': 'numeric'},
        {'name': 'cancel_like_user_num', 'type': 'numeric'},
        {'name': 'comment_cnt', 'type': 'numeric'},
        {'name': 'comment_user_num', 'type': 'numeric'},
        {'name': 'direct_comment_cnt', 'type': 'numeric'},
        {'name': 'reply_comment_cnt', 'type': 'numeric'},
        {'name': 'delete_comment_cnt', 'type': 'numeric'},
        {'name': 'delete_comment_user_num', 'type': 'numeric'},
        {'name': 'comment_like_cnt', 'type': 'numeric'},
        {'name': 'comment_like_user_num', 'type': 'numeric'},
        {'name': 'follow_cnt', 'type': 'numeric'},
        {'name': 'follow_user_num', 'type': 'numeric'},
        {'name': 'cancel_follow_cnt', 'type': 'numeric'},
        {'name': 'cancel_follow_user_num', 'type': 'numeric'},
        {'name': 'share_cnt', 'type': 'numeric'},
        {'name': 'share_user_num', 'type': 'numeric'},
        {'name': 'download_cnt', 'type': 'numeric'},
        {'name': 'download_user_num', 'type': 'numeric'},
        {'name': 'report_cnt', 'type': 'numeric'},
        {'name': 'report_user_num', 'type': 'numeric'},
        {'name': 'reduce_similar_cnt', 'type': 'numeric'},
        {'name': 'reduce_similar_user_num', 'type': 'numeric'},
        {'name': 'collect_cnt', 'type': 'numeric'},
        {'name': 'collect_user_num', 'type': 'numeric'},
        {'name': 'cancel_collect_cnt', 'type': 'numeric'},
        {'name': 'cancel_collect_user_num', 'type': 'numeric'},
        {'name': 'direct_comment_user_num', 'type': 'numeric'},
        {'name': 'reply_comment_user_num', 'type': 'numeric'},
        {'name': 'share_all_cnt', 'type': 'numeric'},
        {'name': 'share_all_user_num', 'type': 'numeric'},
        {'name': 'outsite_share_all_cnt', 'type': 'numeric'},
    ]

    columns_basic = []
    for entry in feat_meta_basic:
        ft_name = entry['name']
        columns_basic.append(ft_name)
        if 'preprocess' in entry:
            item_features[ft_name] = item_features[ft_name].apply(entry['preprocess'])
        elif entry['type'] == 'categoric' and 'default' in entry:
            item_features[ft_name] = item_features[ft_name].apply(
                lambda x: x if x in entry['values'] else entry['default']
            )
        if entry['type'] == 'categoric' and 'values' in entry:
            item_features[ft_name] = item_features[ft_name].apply(
                lambda x: entry['values'].index(x)
            )

    columns_stats = []
    for entry in feat_meta_stat:
        ft_name = entry['name']
        columns_stats.append(ft_name)
        if entry['type'] == 'numeric':
            col_max = item_statistic_features[ft_name].max()
            item_statistic_features[ft_name] = item_statistic_features[ft_name].apply(
                lambda x: x / col_max
            )

    item_selected_basic_features = item_features[columns_basic]
    item_selected_stats_features = item_statistic_features[columns_stats]

    final = pd.merge(item_selected_basic_features, item_selected_stats_features, on='video_id', how='left')
    final = final.drop(columns=['video_id'])

    meta = feat_meta_basic
    meta.extend(feat_meta_stat[1:])
    meta = pd.DataFrame(meta)[['name', 'type', 'count']]
    return final, meta

def mp_apply_by_group(df, grouping_fn, group_fn):
    groups = grouping_fn(df)
    with mp.Pool() as pool:
        n = len(groups)
        results = tqdm(pool.imap(group_fn, groups),total=n)
        results = pd.concat(list(results), axis=0)
    return results

def p1_positive_behavior_offset_process_group(group):
    group['positive_behavior_offset'] = group['positive_behavior_offset'].ffill()
    return group

def p1_item_positive_behavior_offset_process_group(group):
    group['item_positive_behavior_offset'] = group['item_positive_behavior_offset'].ffill()
    return group

def p1_item_negative_behavior_offset_process_group(group):
    group['item_negative_behavior_offset'] = group['item_negative_behavior_offset'].ffill()
    return group

def p1_group_by_user(df):
    groups = df.groupby(['user_id'], as_index=False)
    groups = [g for k,g in groups]
    return groups

def p1_group_by_item(df):
    groups = df.groupby(['item_id'], as_index=False)
    groups = [g for k,g in groups]
    return groups

def preprocess_interaction_data_part1(data, hot_item_threshold=50, hot_user_threshold=10):
    data_positive = data[data['is_click'] == 1]
    data_negative = data[data['is_click'] == 0]

    user_num = data['user_id'].max() + 1
    item_num = data['item_id'].max() + 1

    item_count = data_positive['item_id'].value_counts()
    item_count = pd.DataFrame({'item_id': item_count.index, 'count': item_count.values})
    hot_items = item_count[item_count['count'] >= hot_item_threshold]
    cold_item_ids = [i for i in range(data['item_id'].max()) if i not in hot_items['item_id'].values]
    hot_item_ids = hot_items['item_id'].values

    user_count = data_positive['user_id'].value_counts()
    user_count = pd.DataFrame({'user_id': user_count.index, 'count': user_count.values})
    hot_users = user_count[user_count['count'] >= hot_user_threshold]
    cold_user_ids = [i for i in range(data['user_id'].max()) if i not in hot_users['user_id'].values]
    hot_user_ids = hot_users['user_id'].values

    data_positive = data_positive.sort_values(['user_id', 'ts'], ascending=[True, True])
    data_positive['positive_behavior_offset'] = data_positive.groupby('user_id').cumcount()
    data_positive['positive_behavior_offset'] = data_positive.groupby(['user_id', 'ts'])['positive_behavior_offset'].transform('min')

    data_positive = data_positive.sort_values(['item_id', 'ts'], ascending=[True, True])
    data_positive['item_positive_behavior_offset'] = data_positive.groupby('item_id').cumcount()
    data_positive['item_positive_behavior_offset'] = data_positive.groupby(['item_id', 'ts'])['item_positive_behavior_offset'].transform('min')

    data_negative = data_negative.sort_values(['item_id', 'ts'], ascending=[True, True])
    data_negative['item_negative_behavior_offset'] = data_negative.groupby('item_id').cumcount()
    data_negative['item_negative_behavior_offset'] = data_negative.groupby(['item_id', 'ts'])['item_negative_behavior_offset'].transform('min')

    data = data.merge(
        data_positive[[
            'user_id', 'item_id', 'ts', 'positive_behavior_offset', 'item_positive_behavior_offset'
        ]], on=[
            'user_id', 'item_id', 'ts'
        ], how='left'
    ).sort_values(
        by=['user_id', 'ts'],
        ascending=[True, True]
    )
    data = mp_apply_by_group(data, p1_group_by_user, p1_positive_behavior_offset_process_group)
    data = data.reset_index(drop=True)

    data = data.sort_values(by=['item_id', 'ts'], ascending=[True, True])
    data = mp_apply_by_group(data, p1_group_by_item, p1_item_positive_behavior_offset_process_group)
    data = data.reset_index(drop=True)

    data = data.merge(data_negative[['item_id', 'ts', 'item_negative_behavior_offset']], on=['item_id', 'ts'], how='left')

    data = data.sort_values(by=['item_id', 'ts'], ascending=[True, True])
    data = mp_apply_by_group(data, p1_group_by_item, p1_item_negative_behavior_offset_process_group)
    data = data.reset_index(drop=True)

    data['label'] = (data['is_click'] == 1).astype(int)

    _cold_item_ids = set(cold_item_ids)
    _cold_user_ids = set(cold_user_ids)
    _hot_item_ids = set(hot_item_ids)
    _hot_user_ids = set(hot_user_ids)

    data['cold_item'] = data['item_id'].isin(_cold_item_ids).astype(int)
    data['cold_user'] = data['user_id'].isin(_cold_user_ids).astype(int)
    data['hot_item'] = data['item_id'].isin(_hot_item_ids).astype(int)
    data['hot_user'] = data['user_id'].isin(_hot_user_ids).astype(int)

    data = data.fillna({
        'positive_behavior_offset': 0,
        'item_positive_behavior_offset': 0,
        'item_negative_behavior_offset': 0
    })

    data['user_id'] = data['user_id'].astype(int)
    data['item_id'] = data['item_id'].astype(int)
    data['positive_behavior_offset'] = data['positive_behavior_offset'].astype(int)
    data['item_positive_behavior_offset'] = data['item_positive_behavior_offset'].astype(int)
    data['item_negative_behavior_offset'] = data['item_negative_behavior_offset'].astype(int)

    return data, user_num, item_num, data_positive, data_negative, cold_item_ids, cold_user_ids, hot_item_ids, hot_user_ids

def p2_get_second_to_last_row(group):
    if len(group) >= 2:
        return group.iloc[-2]
    else:
        return None

def p2_get_last_row(group):
    if len(group) >= 1:
        return group.iloc[-1]
    else:
        return None

def p2_group_by_user(df):
    groups = df.groupby('user_id')
    groups = [g for k,g in groups]
    return groups

def preprocess_interaction_data_part2(data, data_positive, data_negative, hot_user_ids):
    '''
    train, val, test splits
    [train_0, ..., train_n, val, test]
    '''
    _hot_user_ids = set(hot_user_ids)

    tmp_data = data[data['label'] == 1].sort_values(by=['user_id', 'ts'], ascending=[True, True])

    val_data = tmp_data.groupby('user_id').apply(p2_get_second_to_last_row).dropna()
    #val_data = mp_apply_by_group(tmp_data, p2_group_by_user, p2_get_second_to_last_row).dropna()

    test_data = tmp_data.groupby('user_id').apply(p2_get_last_row).dropna()
    #test_data = mp_apply_by_group(tmp_data, p2_group_by_user, p2_get_last_row).dropna()

    del tmp_data
    merged_data = pd.concat([data, val_data, test_data])

    train_data = merged_data.drop_duplicates(
        keep=False
    ).sort_values(
        by=['user_id', 'ts'],
        ascending=[True, True]
    ).query('positive_behavior_offset >= 3')
    del merged_data
    print('train')

    val_data = val_data.query('user_id in @_hot_user_ids')
    print('val')
    test_data = test_data.query('user_id in @_hot_user_ids')
    print('test')

    return train_data, val_data, test_data

def preprocess_interaction_data_part3(user_num, item_num, data_positive, data_negative):
    history = {}
    history_ts = {}
    for val, other, val_num in [
        ('user', 'item', user_num),
        ('item', 'user', item_num),
    ]:
        for pos_neg in ['positive', 'negative']:
            val_id = "{}_id".format(val)
            oth_id = "{}_id".format(other)
            print(val, pos_neg)
            if pos_neg == 'positive':
                data = data_positive.sort_values(by=[val_id, 'ts'], ascending=[True, True])
            else:
                data = data_negative.sort_values(by=[val_id, 'ts'], ascending=[True, True])
            val_history = data.groupby(val_id)[oth_id].apply(list).to_dict()
            val_history_ts = data.groupby(val_id)['ts'].apply(list).to_dict()
            for i in tqdm(range(val_num)):
                if i not in val_history:
                    val_history[i] = []
                    val_history_ts[i] = []
            history[(val, pos_neg)] = val_history
            history_ts[(val, pos_neg)] = val_history_ts

    return history, history_ts

def preprocess_interaction_data_part4(data, item_num, split_data, user_history_positive, item_history_positive, item_history_positive_ts, offset=0, neg_num=100, feedback_max_length=10):

    def random_neq(l, r, s):
        t = np.random.randint(l, r)
        while t in s:
            t = np.random.randint(l, r)
        return t

    def find_largest_index_less_than_target(arr, target):
        left, right = 0, len(arr) - 1
        result = -1

        while left <= right:
            mid = left + (right - left) // 2
            if arr[mid] < target:
                result = mid
                left = mid + 1
            else:
                right = mid - 1

        return result

    data_neg_items = []
    data_neg_item_pos_feedbacks = []
    for idx in tqdm(range(len(split_data))):
        user_id = int(split_data.iloc[idx]['user_id'])
        ts = int(split_data.iloc[idx]['ts'])
        neg_item_ids = []
        raw_neg_item_pos_feedbacks = []
        total_history_items = user_history_positive[user_id]
        if offset > 0:
            total_history_items = total_history_items[:-offset]
        for _ in range(neg_num):
            neg_item_id = random_neq(0, item_num, set(total_history_items + neg_item_ids))
            p = find_largest_index_less_than_target(item_history_positive_ts[neg_item_id], ts)
            raw_neg_item_pos_feedback = item_history_positive[neg_item_id][p+1-feedback_max_length:p+1]

            neg_item_ids.append(neg_item_id)
            raw_neg_item_pos_feedbacks.append(raw_neg_item_pos_feedback)
        data_neg_items.append(neg_item_ids)
        data_neg_item_pos_feedbacks.append(raw_neg_item_pos_feedbacks)

    return data_neg_items, data_neg_item_pos_feedbacks

def preprocess_interaction_data(data, feedback_max_length=10, neg_num=100, hot_item_threshold=50, hot_user_threshold=10):
    print('part1')
    data, user_num, item_num, data_positive, data_negative, cold_item_ids, cold_user_ids, hot_item_ids, hot_user_ids = preprocess_interaction_data_part1(
        data,
        hot_item_threshold=hot_item_threshold,
        hot_user_threshold=hot_user_threshold,
    )

    print('part2')
    train_data, val_data, test_data = preprocess_interaction_data_part2(
        data, data_positive, data_negative, hot_user_ids
    )
    print('part3')
    history, history_ts = preprocess_interaction_data_part3(
        user_num, item_num, data_positive, data_negative
    )
    user_history_positive = history[('user', 'positive')]
    item_history_positive = history[('item', 'positive')]
    user_history_negative = history[('user', 'negative')]
    item_history_negative = history[('item', 'negative')]
    user_history_positive_ts = history_ts[('user', 'positive')]
    item_history_positive_ts = history_ts[('item', 'positive')]
    user_history_negative_ts = history_ts[('user', 'negative')]
    item_history_negative_ts = history_ts[('item', 'negative')]

    np.random.seed(0)
    print('part4')
    val_data_neg_items, val_data_neg_item_pos_feedbacks = preprocess_interaction_data_part4(
        data, item_num, val_data, user_history_positive, item_history_positive, item_history_positive_ts,
        offset=1,
        neg_num=neg_num,
        feedback_max_length=feedback_max_length,
    )
    test_data_neg_items, test_data_neg_item_pos_feedbacks = preprocess_interaction_data_part4(
        data, item_num, test_data, user_history_positive, item_history_positive, item_history_positive_ts,
        neg_num=neg_num,
        feedback_max_length=feedback_max_length,
    )

    return (
        user_num,
        item_num,
        cold_item_ids,
        train_data,
        val_data,
        test_data,
        user_history_positive,
        item_history_positive,
        user_history_negative,
        item_history_negative,
        val_data_neg_items,
        val_data_neg_item_pos_feedbacks,
        test_data_neg_items,
        test_data_neg_item_pos_feedbacks,
    )

def preprocess_interaction_features(ds_name, data):
    neg_num = 100
    feedback_max_length = 10
    if ds_name == 'KuaiRand':
        hot_item_threshold = 50
        hot_user_threshold = 10
        _, _, _, interaction_features = data
        interaction_features = interaction_features[['user_id', 'video_id', 'time_ms', 'is_click']]
        interaction_features.columns = ['user_id', 'item_id', 'ts', 'is_click']
    elif ds_name.startswith('MovieLens'):
        if ds_name == 'MovieLens100k':
            hot_item_threshold = 20
            hot_user_threshold = 10
        elif ds_name == 'MovieLens1m':
            hot_item_threshold = 50
            hot_user_threshold = 10
        click_positive_threshold = 4
        click_negative_threshold = 2

        _, _, interaction_features = data
        interaction_features = interaction_features[['user_id', 'item_id', 'timestamp', 'rating']]
        interaction_features.columns = ['user_id', 'item_id', 'ts', 'is_click']

        def is_click_threshold(x):
            if x <= click_negative_threshold:
                return 0 # negative
            elif x >= click_positive_threshold:
                return 1 # positive
            else:
                return -1 # neutral
        interaction_features['is_click'] = interaction_features['is_click'].apply(is_click_threshold)
    elif ds_name.startswith('Amazon'):
        hot_item_threshold = 50
        hot_user_threshold = 10
        click_positive_threshold = 4
        click_negative_threshold = 2

        _, _, interaction_features = data
        interaction_features = interaction_features[['user_id', 'item_id', 'timestamp', 'rating']]
        interaction_features.columns = ['user_id', 'item_id', 'ts', 'is_click']
        def is_click_threshold(x):
            if x <= click_negative_threshold:
                return 0 # negative
            elif x >= click_positive_threshold:
                return 1 # positive
            else:
                return -1 # neutral
        interaction_features['is_click'] = interaction_features['is_click'].apply(is_click_threshold)

    else:
        raise Exception("Not Implemented")

    (
        user_num,
        item_num,
        cold_item_ids,
        train_data,
        val_data,
        test_data,
        user_history_positive,
        item_history_positive,
        user_history_negative,
        item_history_negative,
        val_data_neg_items,
        val_data_neg_item_pos_feedbacks,
        test_data_neg_items,
        test_data_neg_item_pos_feedbacks,
    ) = preprocess_interaction_data(
        interaction_features,
        neg_num=neg_num,
        feedback_max_length=feedback_max_length,
        hot_item_threshold=hot_item_threshold,
        hot_user_threshold=hot_user_threshold,
    )

    meta_data = {'dataset_name': ds_name, 'user_num': user_num, 'item_num': item_num}
    meta_data = pd.DataFrame(meta_data, index=[0])

    return (
        meta_data,
        cold_item_ids,
        train_data,
        val_data,
        test_data,
        user_history_positive,
        item_history_positive,
        user_history_negative,
        item_history_negative,
        val_data_neg_items,
        val_data_neg_item_pos_feedbacks,
        test_data_neg_items,
        test_data_neg_item_pos_feedbacks,
    )

def preprocess_movielens_item_features(item_features):
    columns = []
    feat_meta = [
        {'name': 'movie_id', 'type': 'id', 'count': item_features.shape[0]},
        #'movie_title', #string
        #'release_date', #string, hasnan
        #'video_release_date', # all nan ?
        #'IMDb_URL', # string, hasnan
        {'name': 'unknown', 'type': 'binary'},
        {'name': 'Action', 'type': 'binary'},
        {'name': 'Adventure', 'type': 'binary'},
        {"name": 'Animation', "type": "binary"},
        {"name": "Children's", "type": "binary"},
        {"name": "Comedy", "type": "binary"},
        {"name": "Crime", "type": "binary"},
        {"name": "Documentary", "type": "binary"},
        {"name": "Drama", "type": "binary"},
        {"name": "Fantasy", "type": "binary"},
        {"name": "Film-Noir", "type": "binary"},
        {"name": "Horror", "type": "binary"},
        {"name": "Musical", "type": "binary"},
        {"name": "Mystery", "type": "binary"},
        {"name": "Romance", "type": "binary"},
        {"name": "Sci-Fi", "type": "binary"},
        {"name": "Thriller", "type": "binary"},
        {"name": "War", "type": "binary"},
        {"name": "Western", "type": "binary"},
    ]
    for entry in feat_meta:
        ft_name = entry['name']
        columns.append(ft_name)

    final = item_features[columns]
    final = final.drop(columns='movie_id')

    meta = pd.DataFrame(feat_meta)[['name', 'type', 'count']]
    return final, meta

def preprocess_movielens_user_features(user_features):
    columns = []
    feat_meta = [
        {'name': 'user_id', 'type': 'id', 'count': user_features.shape[0]},
        {'name': 'gender', 'type': 'categoric', 'count': 2, 'values': ['M', 'F']},
        {'name': 'age', 'type': 'numeric'},
        {'name': 'occupation', 'type': 'categoric', 'count': 21},
    ]
    for entry in feat_meta:
        ft_name = entry['name']
        columns.append(ft_name)
        if 'values' in entry and entry['type'] == 'categoric':
            user_features[ft_name] = user_features[ft_name].apply(
                lambda x: entry['values'].index(x)
            )
        elif entry['type'] == 'id':
            continue
        elif entry['type'] == 'numeric':
            col_max = user_features[ft_name].max()
            user_features[ft_name] = user_features[ft_name].apply(
                lambda x: x / col_max
            )

    final = user_features[columns]
    final = final.drop(columns='user_id')

    meta = pd.DataFrame(feat_meta)[['name', 'type', 'count']]
    return final, meta

def preprocess_amazon_item_features(item_features):
    columns = []
    feat_meta = [
        {'name': 'item_id', 'type': 'id', 'count': item_features.shape[0]},
        {'name': 'price', 'type': 'numeric'},
        {'name': 'has_price', 'type': 'binary'},
        {'name': 'rating_number', 'type': 'numeric'},
        {'name': 'average_rating', 'type': 'numeric'},
    ]
    for entry in feat_meta:
        columns.append(entry['name'])

    item_features['rating_number'] = item_features['rating_number'].fillna(0)
    item_features['average_rating'] = item_features['average_rating'].fillna(0)

    final = item_features[columns]
    meta = pd.DataFrame(feat_meta)[['name', 'type', 'count']]
    return final, meta

def preprocess_amazon_user_features(user_features):
    columns = [
        'dummy_ft'
    ]
    feat_meta = [
        {'name': 'user_id', 'type': 'id', 'count': user_features.shape[0]},
        {'name': 'dummy_ft', 'type': 'binary'},
    ]
    user_features['dummy_ft'] = 1
    final = user_features[columns]
    meta = pd.DataFrame(feat_meta)[['name', 'type', 'count']]
    return final, meta

def preprocess_user_features(ds_name, data):
    if ds_name == 'KuaiRand':
        user_features, _, _, _ = data
        user_features, user_features_meta = preprocess_kuairand_user_features(user_features)
    elif ds_name.startswith('MovieLens'):
        user_features, _, _ = data
        user_features, user_features_meta = preprocess_movielens_user_features(user_features)
    elif ds_name.startswith('Amazon'):
        user_features, _, _ = data
        user_features, user_features_meta = preprocess_amazon_user_features(user_features)
    else:
        raise Exception("Not Implemented")
    return user_features, user_features_meta

def flatten_features(features, metadata):
    out_meta = []

    embedding_features = []
    numeric_features = []
    binary_features = []
    onehot_features = []

    for _, entry in metadata.iterrows():
        ft_name = entry['name']
        ft_type = entry['type']
        ft_count = entry['count']
        if ft_type == 'id':
            continue
        elif ft_type == 'binary' or (ft_type == 'categoric' and ft_count == 2):
            binary_features.append({'name': ft_name, 'type': 'binary'})
        elif ft_type == 'categoric' and ft_count <= 4:
            for i in range(int(ft_count)):
                new_ft_name = '{}_onehot{}'.format(ft_name, i)
                onehot_features.append({'name': new_ft_name, 'type': 'onehot'})
                features.loc[features[ft_name] == i, new_ft_name] = 1
                features[new_ft_name] = features[new_ft_name].fillna(0).astype(int)
        elif ft_type == 'numeric':
            numeric_features.append({'name': ft_name, 'type': 'numeric'})
        elif ft_type == 'categoric':
            out_dim = min(50, int(entry['count'] ** 0.25))
            embedding_features.append({'name': ft_name, 'type': 'embedding', 'in_dim': entry['count'], 'out_dim': out_dim})

    out_meta.extend(embedding_features)
    out_meta.extend([*numeric_features, *binary_features, *onehot_features])
    columns = list(map(lambda x: x['name'], out_meta))
    out_features = features[columns]
    out_meta = pd.DataFrame(out_meta)

    return out_features, out_meta


def preprocess_item_features(ds_name, data):
    if ds_name == 'KuaiRand':
        _, item_features, item_statistic_features, _ = data
        item_features, item_features_meta = preprocess_kuairand_item_features(item_features, item_statistic_features)
    elif ds_name.startswith('MovieLens'):
        _, item_features, _ = data
        item_features, item_features_meta = preprocess_movielens_item_features(item_features)
    elif ds_name.startswith('Amazon'):
        _, item_features, _ = data
        item_features, item_features_meta = preprocess_amazon_item_features(item_features)
    else:
        raise Exception("Not Implemented")

    return item_features, item_features_meta

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='KuaiRand', choices=[
        'KuaiRand',
        'MovieLens100k',
        'MovieLens1m',
        'AmazonGames',
        'AmazonFashion',
    ])
    parser.add_argument('--interactions', action='store_true', default=False)
    parser.add_argument('--items', action='store_true', default=False)
    parser.add_argument('--users', action='store_true', default=False)

    args = parser.parse_args()

    args.output_dir = os.path.join('preprocessed', args.dataset)
    os.makedirs(args.output_dir, exist_ok=True)

    return args

def main():
    args = parse_args()
    data = load_data(args.dataset)
    print('load dataset')

    if args.items:
        print('preprocess item features')
        item_features, item_features_meta = preprocess_item_features(args.dataset, data)
        item_features, item_features_meta = flatten_features(item_features, item_features_meta)
        print(item_features)
        item_features.to_csv(os.path.join(args.output_dir, 'item_features.csv'), index=False)
        print(item_features_meta)
        item_features_meta.to_csv(os.path.join(args.output_dir, 'item_features_meta.csv'), index=False)

    if args.users:
        print('preprocess user features')
        user_features, user_features_meta = preprocess_user_features(args.dataset, data)
        user_features, user_features_meta = flatten_features(user_features, user_features_meta)
        print(user_features)
        user_features.to_csv(os.path.join(args.output_dir, 'user_features.csv'), index=False)
        print(user_features_meta)
        user_features_meta.to_csv(os.path.join(args.output_dir, 'user_features_meta.csv'), index=False)

    if args.interactions:
        print('preprocess interaction features')
        (
            meta_data,
            cold_item_ids,
            train_data,
            val_data,
            test_data,
            user_history_positive,
            item_history_positive,
            user_history_negative,
            item_history_negative,
            val_data_neg_items,
            val_data_neg_item_pos_feedbacks,
            test_data_neg_items,
            test_data_neg_item_pos_feedbacks,
        ) = preprocess_interaction_features(args.dataset, data)

        meta_data.to_csv(os.path.join(args.output_dir, 'meta_data.csv'), index=False)
        train_data.to_csv(os.path.join(args.output_dir, 'train_data.csv'), index=False)
        val_data.to_csv(os.path.join(args.output_dir, 'val_data.csv'), index=False)
        test_data.to_csv(os.path.join(args.output_dir, 'test_data.csv'), index=False)

        pickle.dump(cold_item_ids, open(os.path.join(args.output_dir, 'cold_item_ids.pkl'), 'wb'))
        pickle.dump(user_history_positive, open(os.path.join(args.output_dir, 'user_history_positive.pkl'), 'wb'))
        pickle.dump(user_history_negative, open(os.path.join(args.output_dir, 'user_history_negative.pkl'), 'wb'))
        pickle.dump(item_history_positive, open(os.path.join(args.output_dir, 'item_history_positive.pkl'), 'wb'))
        pickle.dump(item_history_negative, open(os.path.join(args.output_dir, 'item_history_negative.pkl'), 'wb'))
        pickle.dump(val_data_neg_items, open(os.path.join(args.output_dir, 'val_data_neg_items.pkl'), 'wb'))
        pickle.dump(val_data_neg_item_pos_feedbacks, open(os.path.join(args.output_dir, 'val_data_neg_item_pos_feedbacks.pkl'), 'wb'))
        pickle.dump(test_data_neg_items, open(os.path.join(args.output_dir, 'test_data_neg_items.pkl'), 'wb'))
        pickle.dump(test_data_neg_item_pos_feedbacks, open(os.path.join(args.output_dir, 'test_data_neg_item_pos_feedbacks.pkl'), 'wb'))

    print('Done')


if __name__ == '__main__':
    main()
