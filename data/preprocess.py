import os
import pickle
import argparse
import numpy as np
import pandas as pd

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

def load_movielens100k():
    _dir = 'MovieLens100k'
    user_features = 'u.user'
    item_features = 'u.item'
    interaction_features = 'u.data'
    occupations = 'u.occupation'

    occupations = os.path.join(_dir, occupations)
    with open(occupations, 'r') as fp:
        occupations = [x.strip() for x in fp.readlines()]

    def decrement_id(x):
        return x - 1

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

    # transform occupation into onehot
    for occ in occupations:
        user_features[occ] = 0
    for occ in occupations:
        cond = user_features['occupation'] == occ
        user_features[cond, occ] = 1
    # also transform in into ID
    user_features['occupation'] = user_features['occupation'].apply(lambda x: occupations.index(x))

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

def load_data(ds_name):
    if ds_name == 'KuaiRand':
        return load_kuairand()
    elif ds_name == 'MovieLens100k':
        return load_movielens100k()
    else:
        raise Exception("Not Implemented")

def preprocess_kuairand_user_features(user_features):
    columns = []
    columns.extend([
        'user_id',
        'user_active_degree',
        'is_live_streamer',
        'is_video_author',
        'follow_user_num_range',
        'fans_user_num_range',
        'friend_user_num_range',
        'register_days_range',
    ])
    for i in range(18):
        columns.append('onehot_feat{}'.format(i))

    # ft: user active degree
    user_active_degree_select_keys = ['full_active', 'high_active', 'middle_active']
    user_active_degree_idx = ['full_active', 'high_active', 'middle_active', 'low_active']
    user_features['user_active_degree'] = user_features['user_active_degree'].apply(
        lambda x: x if x in user_active_degree_select_keys else 'low_active'
    ).apply(
        lambda x: user_active_degree_idx.index(x)
    )

    # ft: is_live_streamer
    is_live_streamer_idx = [-124, 1] 
    user_features['is_live_streamer'] = user_features['is_live_streamer'].apply(
        lambda x: is_live_streamer_idx.index(x)
    )

    # ft: follow_user_num_range
    follow_user_num_range_idx = ['0', '(0,10]', '(10,50]', '(50,100]',
                                 '(100,150]', '(150,250]', '(250,500]', '500+']
    user_features['follow_user_num_range'] = user_features['follow_user_num_range'].apply(
        lambda x: follow_user_num_range_idx.index(x)
    )

    # ft: fans_user_num_range
    fans_user_num_range_select_keys = ['0', '[1,10)', '[10,100)', '[100,1k)',
                               '[1k,5k)', '[5k,1w)']
    fans_user_num_range_idx = ['0', '[1,10)', '[10,100)', '[100,1k)',
                               '[1k,5k)', '[5k,1w)', '1w+']
    user_features['fans_user_num_range'] = user_features['fans_user_num_range'].apply(
        lambda x: x if x in fans_user_num_range_select_keys else '1w+'
    ).apply(
        lambda x: fans_user_num_range_idx.index(x)
    )

    # ft: friend_user_num_range
    friend_user_num_range_idx = ['0', '[1,5)', '[5,30)', '[30,60)',
                               '[60,120)', '[120,250)', '250+']
    user_features['friend_user_num_range'] = user_features['friend_user_num_range'].apply(
        lambda x: friend_user_num_range_idx.index(x)
    )

    # ft: register_days_range
    register_days_range_select_keys = ['31-60', '61-90', '91-180',
                               '181-365', '366-730', '730+']
    register_days_range_idx = ['30-', '31-60', '61-90', '91-180',
                               '181-365', '366-730', '730+']
    user_features['register_days_range'] = user_features['register_days_range'].apply(
        lambda x: x if x in register_days_range_select_keys else '30-'
    ).apply(
        lambda x: register_days_range_idx.index(x)
    )

    # ft: onehot_feat{}'
    for i in range(18):
        ft_name = 'onehot_feat{}'.format(i)
        user_features[ft_name] = user_features[ft_name].apply(
            lambda x: int(x) if x >= 0 else 0
        )

    final = user_features[columns]
    final = final.drop(columns=['user_id'])
    return final

def preprocess_kuairand_item_features(item_features, item_statistic_features):
    columns_basic = [
        'video_id',
        'video_type',
        'video_duration',
        'music_type',
        'tag'
    ]
    columns_stats = item_statistic_features.columns.tolist()[1:]

    video_type_select_keys = ['NORMAL', 'AD']
    video_type_idx = ['NORMAL', 'AD']
    item_features['video_type'] = item_features['video_type'].apply(
        lambda x: x if x in video_type_select_keys else 'NORMAL'
    ).apply(
        lambda x: video_type_idx.index(x)
    )

    def video_duration_process(x):
        if x < 10000:
            return 0
        elif x < 50000:
            return 1
        elif x < 100000:
            return 2
        else:
            return 3
    item_features['video_duration'] = item_features['video_duration'].apply(video_duration_process)

    music_type_idx = [9.0, 4.0, 8.0, 7.0, 11.0]
    item_features['music_type'] = item_features['music_type'].apply(
        lambda x: music_type_idx.index(x) if x in music_type_idx else 5
    )

    def tag_process(x):
        if isinstance(x, str):
            return int(x.split(',')[0])
        return int(x) if x >= 0 else 0
    item_features['tag'] = item_features['tag'].apply(tag_process)

    item_selected_basic_features = item_features[columns_basic]

    for col_name in columns_stats:
        column_max_value = item_statistic_features[col_name].max()
        item_statistic_features[col_name] = item_statistic_features[col_name].apply(lambda x: x / column_max_value)

    final = pd.merge(item_selected_basic_features, item_statistic_features, on='video_id', how='left')
    final = final.drop(columns=['video_id'])
    return final

def preprocess_interaction_data_part1(data, hot_item_threshold=50, hot_user_threshold=10):
    data_positive = data[data['is_click'] == 1]
    data_negative = data[data['is_click'] == 0]

    user_num = data['user_id'].max()+1
    item_num = data['item_id'].max()+1

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


    data = data.merge(data_positive[['user_id', 'item_id', 'ts', 'positive_behavior_offset', 'item_positive_behavior_offset']], on=['user_id', 'item_id', 'ts'], how='left')

    data = data.sort_values(by=['user_id', 'ts'], ascending=[True, True])
    def positive_behavior_offset_process_group(group):
        group['positive_behavior_offset'] = group['positive_behavior_offset'].ffill()
        return group
    data = data.groupby(['user_id'], as_index=False).apply(positive_behavior_offset_process_group).reset_index(drop=True)
    data['positive_behavior_offset'] = data['positive_behavior_offset'].fillna(0)

    data = data.sort_values(by=['item_id', 'ts'], ascending=[True, True])
    def item_positive_behavior_offset_process_group(group):
        group['item_positive_behavior_offset'] = group['item_positive_behavior_offset'].ffill()
        return group
    data = data.groupby(['item_id'], as_index=False).apply(item_positive_behavior_offset_process_group).reset_index(drop=True)
    data['item_positive_behavior_offset'] = data['item_positive_behavior_offset'].fillna(0)

    data = data.merge(data_negative[['item_id', 'ts', 'item_negative_behavior_offset']], on=['item_id', 'ts'], how='left')
    data = data.sort_values(by=['item_id', 'ts'], ascending=[True, True])
    def item_negative_behavior_offset_process_group(group):
        group['item_negative_behavior_offset'] = group['item_negative_behavior_offset'].ffill()
        return group
    data = data.groupby(['item_id'], as_index=False).apply(item_negative_behavior_offset_process_group).reset_index(drop=True)
    data['item_negative_behavior_offset'] = data['item_negative_behavior_offset'].fillna(0)

    data['label'] = (data['is_click'] == 1).astype(int)

    data['cold_item'] = data['item_id'].isin(cold_item_ids).astype(int)
    data['hot_item'] = data['item_id'].isin(hot_item_ids).astype(int)
    data['hot_user'] = data['item_id'].isin(hot_user_ids).astype(int)
    data['cold_user'] = data['item_id'].isin(cold_user_ids).astype(int)

    data.fillna({'positive_behavior_offset': 0, 'item_positive_behavior_offset': 0, 'item_negative_behavior_offset': 0}, inplace=True)

    data['user_id'] = data['user_id'].astype(int)
    data['item_id'] = data['item_id'].astype(int)
    data['positive_behavior_offset'] = data['positive_behavior_offset'].astype(int)
    data['item_positive_behavior_offset'] = data['item_positive_behavior_offset'].astype(int)
    data['item_negative_behavior_offset'] = data['item_negative_behavior_offset'].astype(int)

    return data, user_num, item_num, data_positive, data_negative, cold_item_ids, cold_user_ids, hot_item_ids, hot_user_ids

def preprocess_interaction_data_part2(data, data_positive, data_negative, cold_item_ids, cold_user_ids, hot_item_ids, hot_user_ids):
    '''
    train, val, test splits
    [train_0, ..., train_n, val, test]
    '''
    def get_second_to_last_row(group):
        if len(group) >= 2:
            return group.iloc[-2]
        else:
            return None

    def get_last_row(group):
        if len(group) >= 1:
            return group.iloc[-1]
        else:
            return None

    tmp_data = data[data['label'] == 1].sort_values(by=['user_id', 'ts'], ascending=[True, True])
    val_data = tmp_data.groupby('user_id').apply(get_second_to_last_row)
    val_data = val_data.dropna()
    test_data = tmp_data.groupby('user_id').apply(get_last_row)
    test_data = test_data.dropna()
    merged_data = pd.concat([data, val_data, test_data])
    train_data = merged_data.drop_duplicates(keep=False)
    train_data = train_data.sort_values(by=['user_id', 'ts'], ascending=[True, True])

    train_data = train_data[train_data['positive_behavior_offset'] >= 3]
    val_data = val_data[val_data['user_id'].isin(hot_user_ids)]
    test_data = test_data[test_data['user_id'].isin(hot_user_ids)]

    return train_data, val_data, test_data

def preprocess_interaction_data_part3(data, item_num, user_num, data_positive, data_negative):
    history = {}
    history_ts = {}
    for pos_neg, data in zip(
        ['positive', 'negative'],
        [data_positive, data_negative]
    ):
        for val, other, val_num in [
            ('user', 'item', user_num),
            ('item', 'user', item_num),
        ]:
            val_id = "{}_id".format(val)
            oth_id = "{}_id".format(other)

            data.sort_values(by=[val_id, 'ts'], ascending=[True, True])
            val_history = data.groupby(val_id)[oth_id].apply(list).to_dict()
            val_history_ts = data.groupby(val_id)['ts'].apply(list).to_dict()
            for i in range(val_num):
                if i not in val_history:
                    val_history[i] = []
                    val_history_ts[i] = []
            history[(val, pos_neg)] = val_history
            history_ts[(val, pos_neg)] = val_history_ts

    return history, history_ts

def preprocess_interaction_data_part4(data, split_data, user_history_positive, item_history_positive, item_history_positive_ts, seed=None, offset=0, neg_num=100, feedback_max_length=10):
    item_num = data['item_id'].max() + 1
    if seed:
        np.random.seed(seed)

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
    for idx in range(len(split_data)):
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

def preprocess_interaction_data(data):
    data, user_num, item_num, data_positive, data_negative, cold_item_ids, cold_user_ids, hot_item_ids, hot_user_ids = preprocess_interaction_data_part1(data)
    train_data, val_data, test_data = preprocess_interaction_data_part2(
        data, data_positive, data_negative, cold_item_ids, cold_user_ids, hot_item_ids, hot_user_ids
    )
    history, history_ts = preprocess_interaction_data_part3(
        data, item_num, user_num, data_positive, data_negative
    )

    user_history_positive = history[('user', 'positive')]
    item_history_positive = history[('item', 'positive')]
    user_history_negative = history[('user', 'negative')]
    item_history_negative = history[('item', 'negative')]
    user_history_positive_ts = history_ts[('user', 'positive')]
    item_history_positive_ts = history_ts[('item', 'positive')]
    user_history_negative_ts = history_ts[('user', 'negative')]
    item_history_negative_ts = history_ts[('item', 'negative')]

    val_data_neg_items, val_data_neg_item_pos_feedbacks = preprocess_interaction_data_part4(
        data, val_data, user_history_positive, item_history_positive, item_history_positive_ts, seed=0, offset=1,
    )
    test_data_neg_items, test_data_neg_item_pos_feedbacks = preprocess_interaction_data_part4(
        data, val_data, user_history_positive, item_history_positive, item_history_positive_ts
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
    if ds_name == 'KuaiRand':
        _, _, _, interaction_features = data
        interaction_features = interaction_features[['user_id', 'video_id', 'time_ms', 'is_click']]
        interaction_features.columns = ['user_id', 'item_id', 'ts', 'is_click']
    elif ds_name == 'MovieLens100k':
        is_click_threshold = 1
        _, _, interaction_features = data
        interaction_features = interaction_features[['user_id', 'item_id', 'timestamp', 'rating']]
        interaction_features.columns = ['user_id', 'item_id', 'ts', 'is_click']
        interaction_features['is_click'] = interaction_features['is_click'].apply(
            lambda x: 0 if x < is_click_threshold else 1
        )
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
    ) = preprocess_interaction_data(interaction_features)

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
    '''
    TODO: WIP
    '''
    columns = []
    columns.extend([
        'movie_id',
        #'movie_title', #string
        #'release_date', #string, hasnan
        #'video_release_date', # all nan ?
        #'IMDb_URL', # string, hasnan
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
    ])

    columns.extend([
        'release_year',
        'release_month'
    ])

    # yyyy-mm-dd
    item_features['release_date'] = pd.to_datetime(item_features['release_date'])
    item_features['release_year'] = item_features['release_date'].dt.year.fillna(0).astype(int)
    item_features['release_month'] = item_features['release_date'].dt.month.fillna(0).astype(int)

    final = item_features[columns]
    final = final.drop(columns='movie_id')
    return final

def preprocess_movielens_user_features(user_features):
    '''
    TODO: WIP
    '''
    columns = []
    columns.extend([
        'user_id',
        'age',
        'gender', #
        'occupation',
        #'zip_code', #
        #'administrator',
        #'artist',
        #'doctor',
        #'educator',
        #'engineer',
        #'entertainment',
        #'executive',
        #'healthcare',
        #'homemaker',
        #'lawyer',
        #'librarian',
        #'marketing',
        #'none',
        #'other',
        #'programmer',
        #'retired',
        #'salesman',
        #'scientist',
        #'student',
        #'technician',
        #'writer',
    ])
    user_features['gender'] = user_features['gender'].apply(
        lambda x: 1 if x == 'M' else 0
    ).astype(int)

    final = user_features[columns]
    final = final.drop(columns='user_id')
    return final

def preprocess_user_features(ds_name, data):
    if ds_name == 'KuaiRand':
        user_features, _, _, _ = data
        return preprocess_kuairand_user_features(user_features)
    elif ds_name == 'MovieLens100k':
        user_features, _, _ = data
        return preprocess_movielens_user_features(user_features)
    else:
        raise Exception("Not Implemented")


def preprocess_item_features(ds_name, data):
    if ds_name == 'KuaiRand':
        _, item_features, item_statistic_features, _ = data
        return preprocess_kuairand_item_features(item_features, item_statistic_features)
    elif ds_name == 'MovieLens100k':
        _, item_features, _ = data
        return preprocess_movielens_item_features(item_features)
    else:
        raise Exception("Not Implemented")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='KuaiRand', choices=['KuaiRand', 'MovieLens100k'])
    parser.add_argument('--preprocess', type=bool, default=False)
    parser.add_argument('--save', type=bool, default=False)

    args = parser.parse_args()
    args.output_dir = os.path.join('preprocessed', args.dataset)
    if args.preprocess and args.save:
        os.makedirs(args.output_dir, exist_ok=True)

    return args

if __name__ == '__main__':
    args = parse_args()
    data = load_data(args.dataset)

    if args.preprocess:
        user_features = preprocess_user_features(args.dataset, data)
        item_features = preprocess_item_features(args.dataset, data)
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

    if args.preprocess and args.save:
        item_features.to_csv(os.path.join(args.output_dir, 'item_features.csv'), index=False)
        user_features.to_csv(os.path.join(args.output_dir, 'user_features.csv'), index=False)

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


