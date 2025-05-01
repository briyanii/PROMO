import copy
import json

meta_data = {}

MovieLens100k_meta_data = {"user_num": 943,
                           "item_num": 1682,
                           "user_feature": {
                               "dim": 23,
                               "nume_feat_idx": [
                                    0 # age
                                ],
                               "cate_id_feat_idx": [
                                    (1,2) # gender
                                ],
                               "cate_one_hot_feat_idx": [
                                    (2, 21) # occupation
                                ] # 2-21 occupation: one hot
                               # 类别特征：id类是(idx, cate_num
                               # one hot类是(start_idx, count)
                            },
                            "item_feature":{
                                "dim": 19,
                                "nume_feat_idx": [],
                                "cate_id_feat_idx": [],
                                "cate_one_hot_feat_idx": [
                                    (0, 19) # genre
                                ] # 0-18 genre: one hot / multi hot
                            }
                           }
meta_data['MovieLens100k'] = MovieLens100k_meta_data

MovieLens1m_meta_data = copy.deepcopy(MovieLens100k_meta_data)
MovieLens1m_meta_data['user_num'] = 6040
MovieLens1m_meta_data['item_num'] = 3952
meta_data['MovieLens1m'] = MovieLens1m_meta_data

KuaiRand_meta_data = {"user_num": 27285,
                         "item_num": 7583,
                         "user_feature": {
                               "dim": 25,
                               "nume_feat_idx": [],
                               "cate_id_feat_idx": [
                                    (0, 4), # user active degree
                                            # is_lowactive_period
                                    (1, 2), # is_livestreamer
                                    (2, 2), # is_video_author
                                    (3, 8), # user_follow_num_range
                                    (4, 7), # fan_user_num_range
                                    (5, 7), # friend_user_num_range
                                    (6, 7), # register_days_range
                                    (7, 2), # onehot_feat0
                                    (8, 7), # onehot_feat1
                                    (9, 50),
                                    (10, 1471),
                                    (11, 15),
                                    (12, 34),
                                    (13, 3),
                                    (14, 118),
                                    (15, 454),
                                    (16, 7),
                                    (17, 5),
                                    (18, 5),
                                    (19, 2),
                                    (20, 2),
                                    (21, 2),
                                    (22, 2),
                                    (23, 2),
                                    (24, 2) # onehot_feat17
                                ], #
                               "cate_one_hot_feat_idx": []
                               # 类别特征：id类是(idx, cate_num)，one hot类是(start_idx, count)
                         },
                         "item_feature":{
                            "dim": 55,
                            "nume_feat_idx": [ # all statistic features excluding video id
                                4, # counts
                                5,
                                6,
                                7,
                                8,
                                9,
                                10,
                                11,
                                12,
                                13,
                                14,
                                15,
                                16,
                                17,
                                18,
                                19,
                                20,
                                21,
                                22,
                                23,
                                24,
                                25,
                                26,
                                27,
                                28,
                                29,
                                30,
                                31,
                                32,
                                33,
                                34,
                                35,
                                36,
                                37,
                                38,
                                39,
                                40,
                                41,
                                42,
                                43,
                                44,
                                45,
                                46,
                                47,
                                48,
                                49,
                                50,
                                51,
                                52,
                                53,
                                54
                            ],
                            "cate_id_feat_idx": [
                                    (0,2), # video_type
                                    (1,4), # video_duration (range)
                                    (2,6), # music type
                                    (3,69) # tag
                                ],
                            "cate_one_hot_feat_idx": []
                         }
                        }
meta_data['KuaiRand'] = KuaiRand_meta_data

Tmall_meta_data = {"user_num": 52797,
                         "item_num": 22955,
                         "user_feature": {
                               "dim": 2,
                               "nume_feat_idx": [],
                               "cate_id_feat_idx": [(0, 10), (1, 4)], #
                               "cate_one_hot_feat_idx": []
                               # 类别特征：id类是(idx, cate_num)，one hot类是(start_idx, count)
                         },
                         "item_feature":{
                            "dim": 3,
                            "nume_feat_idx": [],
                            "cate_id_feat_idx": [(0, 774), (1, 3714), (2, 3549)],
                            "cate_one_hot_feat_idx": []
                         }
                        }
meta_data['Tmall'] = Tmall_meta_data

json.dump(meta_data, open('dataset_meta_data.json', 'w'), indent=4)

dataset_meta_data = json.load(open('dataset_meta_data.json', 'r'))
print(dataset_meta_data)
