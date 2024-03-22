import pandas as pd

data = pd.read_csv("./log_standard_4_08_to_4_21_pure.csv")

def user_feature_preprocess():
    user_feature = pd.read_csv("./user_features_pure.csv")
    user_selected_columns = []
    user_selected_columns.append('user_id')

    user_active_degree_dict = user_feature['user_active_degree'].value_counts().to_dict()
    user_active_degree_select_keys = ['full_active', 'high_active', 'middle_active']  
    user_feature['user_active_degree'] = user_feature['user_active_degree'].apply(
        lambda x: x if x in user_active_degree_select_keys else 'low_active')
    user_active_degree_idx = ['full_active', 'high_active', 'middle_active', 'low_active']
    user_feature['user_active_degree'] = user_feature['user_active_degree'].apply(
        lambda x: user_active_degree_idx.index(x))  
    user_selected_columns.append('user_active_degree')

   
    is_live_streamer_idx = [-124, 1] 
    user_feature['is_live_streamer'] = user_feature['is_live_streamer'].apply(lambda x: is_live_streamer_idx.index(x))
    user_selected_columns.append('is_live_streamer')

    user_selected_columns.append('is_video_author')

    follow_user_num_range_dict = user_feature['follow_user_num_range'].value_counts().to_dict()
    follow_user_num_range_idx = ['0', '(0,10]', '(10,50]', '(50,100]',
                                 '(100,150]', '(150,250]', '(250,500]', '500+']
    user_feature['follow_user_num_range'] = user_feature['follow_user_num_range'].apply(
        lambda x: follow_user_num_range_idx.index(x))
    user_selected_columns.append('follow_user_num_range')

    fans_user_num_range_dict = user_feature['fans_user_num_range'].value_counts().to_dict()
    fans_user_num_range_select_keys = ['0', '[1,10)', '[10,100)', '[100,1k)',
                               '[1k,5k)', '[5k,1w)']  
    user_feature['fans_user_num_range'] = user_feature['fans_user_num_range'].apply(
        lambda x: x if x in fans_user_num_range_select_keys else '1w+')
    fans_user_num_range_idx = ['0', '[1,10)', '[10,100)', '[100,1k)',
                               '[1k,5k)', '[5k,1w)', '1w+'] 
    user_feature['fans_user_num_range'] = user_feature['fans_user_num_range'].apply(
        lambda x: fans_user_num_range_idx.index(x))
    user_selected_columns.append('fans_user_num_range')

    friend_user_num_range_dict = user_feature['friend_user_num_range'].value_counts().to_dict()
    friend_user_num_range_idx = ['0', '[1,5)', '[5,30)', '[30,60)',
                               '[60,120)', '[120,250)', '250+'] 
    user_feature['friend_user_num_range'] = user_feature['friend_user_num_range'].apply(
        lambda x: friend_user_num_range_idx.index(x))
    user_selected_columns.append('friend_user_num_range')

    register_days_range_dict = user_feature['register_days_range'].value_counts().to_dict()
    register_days_range_select_keys = ['31-60', '61-90', '91-180',
                               '181-365', '366-730', '730+'] 
    user_feature['register_days_range'] = user_feature['register_days_range'].apply(
        lambda x: x if x in register_days_range_select_keys else '30-')
    register_days_range_idx = ['30-', '31-60', '61-90', '91-180',
                               '181-365', '366-730', '730+'] 
    user_feature['register_days_range'] = user_feature['register_days_range'].apply(
        lambda x: register_days_range_idx.index(x))
    user_selected_columns.append('register_days_range')

    onehot_feat_voc_size = []
    for i in range(18):
        onehot_feat_voc_size.append(int(user_feature['onehot_feat{}'.format(i)].max()+1)) 

        user_feature['onehot_feat{}'.format(i)] = user_feature['onehot_feat{}'.format(i)].apply(
            lambda x: int(x) if x >= 0 else 0)
        user_selected_columns.append('onehot_feat{}'.format(i))

    user_selected_features = user_feature[user_selected_columns]
    user_selected_features.to_csv("./user_selected_features.csv", index=False)

def item_feature_preprocess():
    item_feature = pd.read_csv("./video_features_basic_pure.csv")
    item_selected_columns = []
    item_selected_columns.append('video_id')

    video_type_dict = item_feature['video_type'].value_counts().to_dict()
    video_type_select_keys = ['NORMAL', 'AD']  
    item_feature['video_type'] = item_feature['video_type'].apply(
        lambda x: x if x in video_type_select_keys else 'NORMAL')
    video_type_idx = ['NORMAL', 'AD']  
    item_feature['video_type'] = item_feature['video_type'].apply(lambda x: video_type_idx.index(x))
    item_selected_columns.append('video_type')

    def video_duration_process(x):
        if x < 10000:
            return 0
        elif x < 50000:
            return 1
        elif x < 100000:
            return 2
        else:
            return 3
    item_feature['video_duration'] = item_feature['video_duration'].apply(video_duration_process)
    item_selected_columns.append('video_duration')

    music_type_dict = item_feature['music_type'].value_counts().to_dict()
    music_type_idx = [9.0, 4.0, 8.0, 7.0, 11.0]  
    item_feature['music_type'] = item_feature['music_type'].apply(lambda x: music_type_idx.index(x) if x in music_type_idx else 5)
    item_selected_columns.append('music_type')

    def tag_process(x):
        if isinstance(x, str):
            return int(x.split(',')[0])
        return int(x) if x >= 0 else 0
    item_feature['tag'] = item_feature['tag'].apply(tag_process) 
    item_selected_columns.append('tag')

    item_selected_basic_features = item_feature[item_selected_columns]

    item_statistic_features = pd.read_csv("./video_features_statistic_pure.csv")
    item_statistic_features_columns = item_statistic_features.columns.tolist()
    for columns in item_statistic_features_columns[1:]:
        column_max_value = item_statistic_features[columns].max()
        item_statistic_features[columns] = item_statistic_features[columns].apply(lambda x: x / column_max_value)

    item_selected_features = pd.merge(item_selected_basic_features, item_statistic_features, on='video_id', how='left')
    item_selected_features.to_csv("./item_selected_features.csv", index=False)

user_feature_preprocess()
item_feature_preprocess()


