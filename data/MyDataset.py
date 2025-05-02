import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
import random


def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


class SequentialRecommendationDataset(Dataset):
    def __init__(self, data_dir, max_length=50, mode='train', neg_num=1, device='cpu'):
        self.mode = mode
        self.max_length = max_length
        self.neg_num = neg_num
        self.device = device

        # 从CSV文件中加载meta data
        meta_data = pd.read_csv(data_dir + '/meta_data.csv')
        self.name = meta_data['dataset_name'].values[0]
        self.user_num = meta_data['user_num'].values[0]
        self.item_num = meta_data['item_num'].values[0]

        # 加载用户和物品的特征
        self.user_features = pd.read_csv(data_dir + '/user_features.csv')
        self.user_features_meta = pd.read_csv(data_dir + '/user_features_meta.csv')
        self.user_features_dim = self.user_features.shape[1]
        self.item_features = pd.read_csv(data_dir + '/item_features.csv')
        self.item_features_meta = pd.read_csv(data_dir + '/item_features_meta.csv')
        self.item_features_dim = self.item_features.shape[1]

        # 加载交互数据
        raw_data = pd.read_csv(data_dir + '/data.csv')
        self.user_history = pd.read_pickle(data_dir + '/user_history.pkl')
        self.item_history = pd.read_pickle(data_dir + '/item_history.pkl')
        self.data = range(self.user_num)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取指定索引处的数据
        user_id = self.data[idx]
        if self.mode == 'train':
            target_item_id = self.user_history[user_id][-3]
            history_items = self.user_history[user_id][-(self.max_length+3):-3]
        elif self.mode == 'val':
            target_item_id = self.user_history[user_id][-2]
            history_items = self.user_history[user_id][-(self.max_length+2):-2]
        elif self.mode == 'test':
            target_item_id = self.user_history[user_id][-1]
            history_items = self.user_history[user_id][-(self.max_length+1):-1]
        else:
            raise ValueError('mode must be train/val/test')

        neg_item_ids = []
        for _ in range(self.neg_num):
            neg_item_id = random_neq(0, self.item_num, set(history_items+[target_item_id]+neg_item_ids))
            neg_item_ids.append(neg_item_id)

        # 获取用户和物品的特征
        user_features = self.user_features.iloc[user_id]
        item_features = self.item_features.iloc[target_item_id]
        neg_item_features = self.item_features.iloc[neg_item_ids]

        # 转化成tensor
        user_id = torch.LongTensor([user_id]).to(self.device)
        history_items = torch.LongTensor(history_items + [0] * (self.max_length-len(history_items))).to(self.device)
        history_items_len = torch.LongTensor([1] * len(history_items) + [0] * (self.max_length-len(history_items))).to(self.device)
        target_item_id = torch.LongTensor([target_item_id]).to(self.device)
        neg_item_id = torch.LongTensor(neg_item_ids).to(self.device)
        user_features = torch.FloatTensor(user_features.values).to(self.device)
        item_features = torch.FloatTensor(item_features.values).to(self.device)
        neg_item_features = torch.FloatTensor(neg_item_features.values).to(self.device)

        # 返回样本
        return user_id, history_items, history_items_len, \
            target_item_id, neg_item_id, \
            user_features, item_features, neg_item_features


class MyDataset(Dataset):
    def __init__(self, data_dir, max_length=50, mode='train', neg_num=1, device='cpu'):
        self.mode = mode
        self.max_length = max_length
        self.neg_num = PTCRDataset
        self.device = device

        # 从CSV文件中加载meta data
        meta_data = pd.read_csv(data_dir + '/meta_data.csv')
        self.name = meta_data['dataset_name'].values[0]
        self.user_num = meta_data['user_num'].values[0]
        self.item_num = meta_data['item_num'].values[0]

        # 加载用户和物品的特征
        self.user_features = pd.read_csv(data_dir + '/user_features.csv')
        self.user_features_meta = pd.read_csv(data_dir + '/user_features_meta.csv')
        self.user_features_dim = self.user_features.shape[1]
        self.item_features = pd.read_csv(data_dir + '/item_features.csv')
        self.item_features_meta = pd.read_csv(data_dir + '/item_features_meta.csv')
        self.item_features_dim = self.item_features.shape[1]

        # 加载交互数据
        self.data = pd.read_csv(data_dir + '/' + mode + '_data.csv')
        # self.user_history = pd.read_pickle(data_dir + '/user_history.pkl')
        # self.item_history = pd.read_pickle(data_dir + '/item_history.pkl')
        self.user_history_positive = pd.read_pickle(data_dir + '/user_history_positive.pkl')
        self.user_history_negative = pd.read_pickle(data_dir + '/user_history_negative.pkl')
        self.item_history_positive = pd.read_pickle(data_dir + '/item_history_positive.pkl')
        self.item_history_negative = pd.read_pickle(data_dir + '/item_history_negative.pkl')

        # 如果是test或val，则为每个样本生成negative item, 固定为100个
        if self.mode == 'val' or self.mode == 'test':
            self.data_neg_items = pd.read_pickle(data_dir + '/' + mode + '_data_neg_items.pkl')


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取指定索引处的数据
        row = self.data.iloc[idx]
        user_id = int(row['user_id'])
        target_item_id = int(row['item_id'])
        positive_behavior_offset = int(row['positive_behavior_offset'])
        label = row['label']
        cold_item = row['cold_item']

        raw_history_items = self.user_history_positive[user_id][
                        max(0, positive_behavior_offset + 1 - self.max_length):positive_behavior_offset + 1]
        user_features = self.user_features.iloc[user_id]
        item_features = self.item_features.iloc[target_item_id]

        if self.mode == 'train':
            # 转化成tensor
            user_id = torch.LongTensor([user_id]).to(self.device)
            history_items = torch.LongTensor(
                raw_history_items
                + [0] * (self.max_length - len(raw_history_items))
            ).to(self.device)
            history_items_len = torch.LongTensor(
                [1] * len(raw_history_items)
                + [0] * (self.max_length - len(raw_history_items))
            ).to(self.device)
            target_item_id = torch.LongTensor([target_item_id]).to(self.device)
            user_features = torch.FloatTensor(user_features.values).to(self.device)
            item_features = torch.FloatTensor(item_features.values).to(self.device)
            label = torch.FloatTensor([label]).to(self.device)
            cold_item = torch.FloatTensor([cold_item]).to(self.device)

            return user_id, history_items, history_items_len, target_item_id,\
                user_features, item_features, label, cold_item

        elif self.mode == 'val' or self.mode == 'test':
            neg_item_ids = self.data_neg_items[idx]
            neg_item_features = self.item_features.iloc[neg_item_ids]

            # 转化成tensor
            user_id = torch.LongTensor([user_id]).to(self.device)
            history_items = torch.LongTensor(
                raw_history_items
                + [0] * (self.max_length - len(raw_history_items))
            ).to(self.device)
            history_items_len = torch.LongTensor(
                [1] * len(raw_history_items)
                + [0] * (self.max_length - len(raw_history_items))
            ).to(self.device)
            target_item_id = torch.LongTensor([target_item_id]).to(self.device)
            neg_item_id = torch.LongTensor(neg_item_ids).to(self.device)
            user_features = torch.FloatTensor(user_features.values).to(self.device)
            item_features = torch.FloatTensor(item_features.values).to(self.device)
            neg_item_features = torch.FloatTensor(neg_item_features.values).to(self.device)

            # 返回样本
            return user_id, history_items, history_items_len, \
                target_item_id, neg_item_id, \
                user_features, item_features, neg_item_features

        else:
            raise ValueError('mode must be train/val/test')


class PTCRDataset(Dataset):
    def __init__(self, data_dir, max_length=50, feedback_max_length=10, mode='train', neg_num=1, device='cpu'):
        self.mode = mode
        self.max_length = max_length
        self.feedback_max_length = feedback_max_length
        self.neg_num = neg_num
        self.device = device

        # 从CSV文件中加载meta data
        meta_data = pd.read_csv(data_dir + '/meta_data.csv')
        self.name = meta_data['dataset_name'].values[0]
        self.user_num = meta_data['user_num'].values[0]
        self.item_num = meta_data['item_num'].values[0]

        # 加载用户和物品的特征
        self.user_features = pd.read_csv(data_dir + '/user_features.csv')
        self.user_features_meta = pd.read_csv(data_dir + '/user_features_meta.csv')
        self.user_features_dim = self.user_features.shape[1]
        self.item_features = pd.read_csv(data_dir + '/item_features.csv')
        self.item_features_meta = pd.read_csv(data_dir + '/item_features_meta.csv')
        self.item_features_dim = self.item_features.shape[1]

        # 加载交互数据
        self.data = pd.read_csv(data_dir + '/' + mode + '_data.csv')
        # self.user_history = pd.read_pickle(data_dir + '/user_history.pkl')
        # self.item_history = pd.read_pickle(data_dir + '/item_history.pkl')
        self.user_history_positive = pd.read_pickle(data_dir + '/user_history_positive.pkl')
        self.user_history_negative = pd.read_pickle(data_dir + '/user_history_negative.pkl')
        self.item_history_positive = pd.read_pickle(data_dir + '/item_history_positive.pkl')
        self.item_history_negative = pd.read_pickle(data_dir + '/item_history_negative.pkl')

        # 如果是test或val，则为每个样本生成negative item, 固定为100个
        if self.mode == 'val' or self.mode == 'test':
            self.data_neg_items = pd.read_pickle(data_dir + '/' + mode + '_data_neg_items.pkl')
            raw_data_neg_item_pos_feedbacks = pd.read_pickle(data_dir + '/' + mode + '_data_neg_item_pos_feedbacks.pkl')
            self.data_neg_item_pos_feedbacks = []
            self.data_neg_item_pos_feedback_lens = []

            for i in range(len(self.data)):
                neg_item_pos_feedbacks = []
                neg_item_pos_feedback_lens = []
                for j in range(self.neg_num):
                    raw_neg_item_pos_feedback = raw_data_neg_item_pos_feedbacks[i][j]
                    neg_item_pos_feedback = torch.LongTensor(
                        raw_neg_item_pos_feedback 
                        + [0] * (self.feedback_max_length - len(raw_neg_item_pos_feedback))
                    )
                    neg_item_pos_feedback_len = torch.LongTensor(
                        [1] * len(raw_neg_item_pos_feedback) 
                        + [0] * (self.feedback_max_length - len(raw_neg_item_pos_feedback))
                    )
                    neg_item_pos_feedbacks.append(neg_item_pos_feedback)
                    neg_item_pos_feedback_lens.append(neg_item_pos_feedback_len)
                self.data_neg_item_pos_feedbacks.append(neg_item_pos_feedbacks)
                self.data_neg_item_pos_feedback_lens.append(neg_item_pos_feedback_lens)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取指定索引处的数据
        row = self.data.iloc[idx]
        user_id = int(row['user_id'])
        target_item_id = int(row['item_id'])
        positive_behavior_offset = int(row['positive_behavior_offset'])
        item_positive_behavior_offset = int(row['item_positive_behavior_offset'])
        item_negative_behavior_offset = int(row['item_negative_behavior_offset'])

        label = row['label']
        cold_item = row['cold_item']
        start_idx = max(0, positive_behavior_offset + 1 - self.max_length)
        end_idx = positive_behavior_offset + 1
        raw_history_items = self.user_history_positive[user_id][start_idx:end_idx]
        user_features = self.user_features.iloc[user_id]
        item_features = self.item_features.iloc[target_item_id]

        user_id = torch.LongTensor([user_id]).to(self.device)
        history_items = torch.LongTensor(
            raw_history_items 
            + [0] * (self.max_length - len(raw_history_items))
        ).to(self.device)
        history_items_len = torch.LongTensor(
            [1] * len(raw_history_items) 
            + [0] * (self.max_length - len(raw_history_items))
        ).to(self.device)

        start_idx = max(0, item_positive_behavior_offset + 1 - self.feedback_max_length)
        end_idx = item_positive_behavior_offset + 1
        raw_item_pos_feedback = self.item_history_positive[target_item_id][start_idx:end_idx]

        start_idx = max(0, item_negative_behavior_offset + 1 - self.feedback_max_length)
        end_idx = item_negative_behavior_offset + 1
        raw_item_neg_feedback = self.item_history_negative[target_item_id][start_idx:end_idx]

        target_item_id = torch.LongTensor([target_item_id]).to(self.device)
        user_features = torch.FloatTensor(user_features.values).to(self.device)
        item_features = torch.FloatTensor(item_features.values).to(self.device)

        item_pos_feedback = torch.LongTensor(
            raw_item_pos_feedback 
            + [0] * (self.feedback_max_length - len(raw_item_pos_feedback))
        ).to(self.device)
        item_pos_feedback_len = torch.LongTensor(
            [1] * len(raw_item_pos_feedback) 
            + [0] * (self.feedback_max_length - len(raw_item_pos_feedback))
        ).to(self.device)

        if self.mode == 'train':
            # 转化成tensor
            label = torch.FloatTensor([label]).to(self.device)
            cold_item = torch.FloatTensor([cold_item]).to(self.device)

            item_neg_feedback = torch.LongTensor(
                raw_item_neg_feedback 
                + [0] * (self.feedback_max_length - len(raw_item_neg_feedback))
            ).to(self.device)
            item_neg_feedback_len = torch.LongTensor(
                [1] * len(raw_item_neg_feedback) 
                + [0] * (self.feedback_max_length - len(raw_item_neg_feedback))
            ).to(self.device)

            return user_id, history_items, history_items_len, target_item_id,\
                user_features, item_features, label, cold_item, \
                item_pos_feedback, item_pos_feedback_len, item_neg_feedback, item_neg_feedback_len

        elif self.mode == 'val' or self.mode == 'test':
            neg_item_ids = self.data_neg_items[idx]
            neg_item_pos_feedbacks = self.data_neg_item_pos_feedbacks[idx]
            neg_item_pos_feedback_lens = self.data_neg_item_pos_feedback_lens[idx]

            neg_item_features = self.item_features.iloc[neg_item_ids]
            neg_item_pos_feedbacks = torch.stack(neg_item_pos_feedbacks, dim=0).to(
                self.device
            ) # [neg_num, feedback_max_length]
            neg_item_pos_feedback_lens = torch.stack(neg_item_pos_feedback_lens, dim=0).to(
                self.device
            ) # [neg_num, feedback_max_length]

            neg_item_id = torch.LongTensor(neg_item_ids).to(self.device)
            neg_item_features = torch.FloatTensor(neg_item_features.values).to(self.device)

            # 返回样本
            return user_id, history_items, history_items_len, \
                target_item_id, neg_item_id, \
                user_features, item_features, neg_item_features, \
                item_pos_feedback, item_pos_feedback_len, neg_item_pos_feedbacks, neg_item_pos_feedback_lens

        else:
            raise ValueError('mode must be train/val/test')


class PLATEDataset(Dataset):
    def __init__(self, data_dir, max_length=50, mode='train', neg_num=1, device='cpu'):
        self.mode = mode
        self.max_length = max_length
        self.neg_num = neg_num
        self.device = device

        # 从CSV文件中加载meta data
        meta_data = pd.read_csv(data_dir + '/meta_data.csv')
        self.name = meta_data['dataset_name'].values[0]
        self.user_num = meta_data['user_num'].values[0]
        self.item_num = meta_data['item_num'].values[0]

        # 加载用户和物品的特征
        self.user_features = pd.read_csv(data_dir + '/user_features.csv')
        self.user_features_meta = pd.read_csv(data_dir + '/user_features_meta.csv')
        self.user_features_dim = self.user_features.shape[1]
        self.item_features = pd.read_csv(data_dir + '/item_features.csv')
        self.item_features_meta = pd.read_csv(data_dir + '/item_features_meta.csv')
        self.item_features_dim = self.item_features.shape[1]

        # 加载交互数据
        self.data = pd.read_csv(data_dir + '/' + mode + '_data.csv')
        # self.user_history = pd.read_pickle(data_dir + '/user_history.pkl')
        # self.item_history = pd.read_pickle(data_dir + '/item_history.pkl')
        self.user_history_positive = pd.read_pickle(data_dir + '/user_history_positive.pkl')
        self.user_history_negative = pd.read_pickle(data_dir + '/user_history_negative.pkl')
        self.item_history_positive = pd.read_pickle(data_dir + '/item_history_positive.pkl')
        self.item_history_negative = pd.read_pickle(data_dir + '/item_history_negative.pkl')

        # 如果是test或val，则为每个样本生成negative item, 固定为100个
        if self.mode == 'val' or self.mode == 'test':
            self.data_neg_items = pd.read_pickle(data_dir + '/' + mode + '_data_neg_items.pkl')

        # 加载冷启动item信息
        self.cold_item_ids = pd.read_pickle(data_dir + '/cold_item_ids.pkl')
        self.cold_item_ids_tensor = torch.zeros(self.item_num, dtype=torch.long)
        self.cold_item_ids_tensor[self.cold_item_ids] = 1


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取指定索引处的数据
        row = self.data.iloc[idx]
        user_id = int(row['user_id'])
        target_item_id = int(row['item_id'])
        positive_behavior_offset = int(row['positive_behavior_offset'])
        label = row['label']
        cold_item = row['cold_item']

        raw_history_items = self.user_history_positive[user_id][
                        max(0, positive_behavior_offset + 1 - self.max_length):positive_behavior_offset + 1]
        user_features = self.user_features.iloc[user_id]
        item_features = self.item_features.iloc[target_item_id]

        if self.mode == 'train':
            # 转化成tensor
            user_id = torch.LongTensor([user_id]).to(self.device)
            history_items = torch.LongTensor(raw_history_items + [0] * (self.max_length - len(raw_history_items))).to(
                self.device)
            history_items_len = torch.LongTensor(
                [1] * len(raw_history_items) + [0] * (self.max_length - len(raw_history_items))).to(self.device)
            target_item_id = torch.LongTensor([target_item_id]).to(self.device)
            user_features = torch.FloatTensor(user_features.values).to(self.device)
            item_features = torch.FloatTensor(item_features.values).to(self.device)
            label = torch.FloatTensor([label]).to(self.device)
            cold_item = torch.LongTensor([cold_item]).to(self.device)

            return user_id, history_items, history_items_len, target_item_id,\
                user_features, item_features, label, cold_item

        elif self.mode == 'val' or self.mode == 'test':
            # neg_item_ids = []
            # if self.mode == 'val':
            #     total_history_items = self.user_history_positive[user_id][:-1]
            # else:
            #     total_history_items = self.user_history_positive[user_id]
            # for _ in range(self.neg_num):
            #     neg_item_id = random_neq(0, self.item_num, set(total_history_items + neg_item_ids))
            #     neg_item_ids.append(neg_item_id)
            neg_item_ids = self.data_neg_items[idx]
            neg_item_features = self.item_features.iloc[neg_item_ids]

            # 转化成tensor
            user_id = torch.LongTensor([user_id]).to(self.device)
            history_items = torch.LongTensor(raw_history_items + [0] * (self.max_length - len(raw_history_items))).to(
                self.device)
            history_items_len = torch.LongTensor(
                [1] * len(raw_history_items) + [0] * (self.max_length - len(raw_history_items))).to(self.device)
            target_item_id = torch.LongTensor([target_item_id]).to(self.device)
            neg_item_id = torch.LongTensor(neg_item_ids).to(self.device)
            user_features = torch.FloatTensor(user_features.values).to(self.device)
            item_features = torch.FloatTensor(item_features.values).to(self.device)
            neg_item_features = torch.FloatTensor(neg_item_features.values).to(self.device)

            # is_cold_items = torch.LongTensor([1 if target_item_id in self.cold_item_ids else 0] +
            #                                        [1 if neg_item_id in self.cold_item_ids else 0
            #                                         for neg_item_id in neg_item_ids]).to(self.device)
            is_cold_items = self.cold_item_ids_tensor[[target_item_id] + neg_item_ids]
            is_cold_items = is_cold_items.unsqueeze(1).to(self.device)

            # 返回样本
            return user_id, history_items, history_items_len, \
                target_item_id, neg_item_id, \
                user_features, item_features, neg_item_features,\
                is_cold_items
        else:
            raise ValueError('mode must be train/val/test')


class MetaEmbDataset(Dataset):
    def __init__(self, data_dir, max_length=50, mode='train', neg_num=1, device='cpu', K=5):
        self.mode = mode
        self.max_length = max_length
        self.neg_num = neg_num
        self.device = device
        self.K = K

        # 从CSV文件中加载meta data
        meta_data = pd.read_csv(data_dir + '/meta_data.csv')
        self.name = meta_data['dataset_name'].values[0]
        self.user_num = meta_data['user_num'].values[0]
        self.item_num = meta_data['item_num'].values[0]

        # 加载用户和物品的特征
        self.user_features = pd.read_csv(data_dir + '/user_features.csv')
        self.user_features_meta = pd.read_csv(data_dir + '/user_features_meta.csv')
        self.user_features_dim = self.user_features.shape[1]
        self.item_features = pd.read_csv(data_dir + '/item_features.csv')
        self.item_features_meta = pd.read_csv(data_dir + '/item_features_meta.csv')
        self.item_features_dim = self.item_features.shape[1]

        # 加载交互数据
        self.data = pd.read_csv(data_dir + '/' + mode + '_data.csv')
        # self.user_history = pd.read_pickle(data_dir + '/user_history.pkl')
        # self.item_history = pd.read_pickle(data_dir + '/item_history.pkl')
        self.user_history_positive = pd.read_pickle(data_dir + '/user_history_positive.pkl')
        self.user_history_negative = pd.read_pickle(data_dir + '/user_history_negative.pkl')
        self.item_history_positive = pd.read_pickle(data_dir + '/item_history_positive.pkl')
        self.item_history_negative = pd.read_pickle(data_dir + '/item_history_negative.pkl')

        # 如果是test或val，则为每个样本生成negative item, 固定为100个
        if self.mode == 'val' or self.mode == 'test':
            self.data_neg_items = pd.read_pickle(data_dir + '/' + mode + '_data_neg_items.pkl')

        self.cold_item_ids = pd.read_pickle(data_dir + '/cold_item_ids.pkl')
        self.hot_item_ids = [i for i in range(self.item_num) if i not in self.cold_item_ids]
        self.cold_item_ids_tensor = torch.zeros(self.item_num, dtype=torch.long)
        self.cold_item_ids_tensor[self.cold_item_ids] = 1


    def __len__(self):
        if self.mode == 'train':
            return len(self.hot_item_ids)
        else:
            return len(self.data)

    def __getitem__(self, idx):
        if self.mode == 'train':
            # 转化成tensor
            target_item_id = self.hot_item_ids[idx]
            pos_user_list = self.item_history_positive[target_item_id]
            neg_user_list = self.item_history_negative[target_item_id]

            user_list = pos_user_list + neg_user_list
            total_label = [1] * len(pos_user_list) + [0] * len(neg_user_list)
            idx_list = list(range(len(user_list)))

            idx_a = random.sample(idx_list, min(self.K, len(user_list)))
            idx_b = random.sample(idx_list, min(self.K, len(user_list)))
            if self.K > len(user_list):
                idx_a += random.choices(idx_list, k=self.K - len(user_list))
                idx_b += random.choices(idx_list, k=self.K - len(user_list))

            user_a = [user_list[i] for i in idx_a]
            user_b = [user_list[i] for i in idx_b]

            user_a_features = self.user_features.iloc[user_a]
            user_b_features = self.user_features.iloc[user_b]
            item_features = self.item_features.iloc[target_item_id]
            history_items_a = []
            history_items_len_a = []
            for user in user_a:
                raw_history_item = self.user_history_positive[user][-self.max_length:]
                history_items_a.append(raw_history_item + [0] * (self.max_length - len(raw_history_item)))
                history_items_len_a.append(
                    [1] * len(raw_history_item) + [0] * (self.max_length - len(raw_history_item)))
            history_items_b = []
            history_items_len_b = []
            for user in user_b:
                raw_history_item = self.user_history_positive[user][-self.max_length:]
                history_items_b.append(raw_history_item + [0] * (self.max_length - len(raw_history_item)))
                history_items_len_b.append(
                    [1] * len(raw_history_item) + [0] * (self.max_length - len(raw_history_item)))

            user_a_ids = torch.LongTensor(user_a).to(self.device)
            user_b_ids = torch.LongTensor(user_b).to(self.device)

            history_items_a = torch.LongTensor(history_items_a).to(self.device)
            history_items_len_a = torch.LongTensor(history_items_len_a).to(self.device)
            history_items_b = torch.LongTensor(history_items_b).to(self.device)
            history_items_len_b = torch.LongTensor(history_items_len_b).to(self.device)
            # history_items_a = torch.LongTensor(raw_history_items_a + [0] * (self.max_length - len(raw_history_items_a))).to(
            #     self.device)
            # history_items_len_a = torch.LongTensor(
            #     [1] * len(raw_history_items_a) + [0] * (self.max_length - len(raw_history_items_a))).to(self.device)
            # history_items_b = torch.LongTensor(raw_history_items_b + [0] * (self.max_length - len(raw_history_items_b))).to(
            #     self.device)
            # history_items_len_b = torch.LongTensor(
            #     [1] * len(raw_history_items_b) + [0] * (self.max_length - len(raw_history_items))).to(self.device)
            target_item_ids = torch.LongTensor([target_item_id]).to(self.device)
            user_a_features = torch.FloatTensor(user_a_features.values).to(self.device)
            user_b_features = torch.FloatTensor(user_b_features.values).to(self.device)
            item_features = torch.FloatTensor(item_features.values).to(self.device)
            label_a = torch.FloatTensor([total_label[i] for i in idx_a]).to(self.device)
            label_b = torch.FloatTensor([total_label[i] for i in idx_b]).to(self.device)
            cold_item = torch.FloatTensor([0]).to(self.device)

            return user_a_ids, user_b_ids, history_items_a, history_items_len_a, history_items_b, history_items_len_b,\
                target_item_ids, user_a_features, user_b_features, item_features, label_a, label_b, cold_item

        elif self.mode == 'val' or self.mode == 'test':
            row = self.data.iloc[idx]
            user_id = int(row['user_id'])
            target_item_id = int(row['item_id'])
            positive_behavior_offset = int(row['positive_behavior_offset'])
            label = row['label']
            cold_item = row['cold_item']

            raw_history_items = self.user_history_positive[user_id][
                                max(0, positive_behavior_offset + 1 - self.max_length):positive_behavior_offset + 1]
            user_features = self.user_features.iloc[user_id]
            item_features = self.item_features.iloc[target_item_id]

            neg_item_ids = self.data_neg_items[idx]
            neg_item_features = self.item_features.iloc[neg_item_ids]

            # 转化成tensor
            user_id = torch.LongTensor([user_id]).to(self.device)
            history_items = torch.LongTensor(raw_history_items + [0] * (self.max_length - len(raw_history_items))).to(
                self.device)
            history_items_len = torch.LongTensor(
                [1] * len(raw_history_items) + [0] * (self.max_length - len(raw_history_items))).to(self.device)
            target_item_id = torch.LongTensor([target_item_id]).to(self.device)
            neg_item_id = torch.LongTensor(neg_item_ids).to(self.device)
            user_features = torch.FloatTensor(user_features.values).to(self.device)
            item_features = torch.FloatTensor(item_features.values).to(self.device)
            neg_item_features = torch.FloatTensor(neg_item_features.values).to(self.device)

            # is_cold_items = torch.LongTensor([1 if target_item_id in self.cold_item_ids else 0] +
            #                                        [1 if neg_item_id in self.cold_item_ids else 0
            #                                         for neg_item_id in neg_item_ids]).to(self.device)
            is_cold_items = self.cold_item_ids_tensor[[target_item_id] + neg_item_ids]
            is_cold_items = is_cold_items.unsqueeze(1).to(self.device)

            # 返回样本
            return user_id, history_items, history_items_len, \
                target_item_id, neg_item_id, \
                user_features, item_features, neg_item_features, \
                is_cold_items

        else:
            raise ValueError('mode must be train/val/test')
