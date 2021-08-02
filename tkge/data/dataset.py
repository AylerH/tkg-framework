import torch
from torch.utils.data.dataset import Dataset as PTDataset
import numpy as np

from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from tkge.common.registrable import Registrable
from tkge.common.configurable import Configurable
from tkge.common.config import Config
from tkge.common.error import ConfigurationError
from tkge.data.utils import get_all_days_of_year

import enum
import arrow
import pdb

from abc import ABC, abstractmethod

SPOT = enum.Enum('spot', ('s', 'p', 'o', 't'))


class DatasetProcessor(ABC, Registrable, Configurable):
    def __init__(self, config: Config):
        Registrable.__init__(self)
        Configurable.__init__(self, config=config)

        self.folder = self.config.get("dataset.folder")
        self.resolution = self.config.get("dataset.temporal.resolution")
        self.index = self.config.get("dataset.temporal.index")
        self.float = self.config.get("dataset.temporal.float")
        self.name = self.config.get("dataset.name")

        self.reciprocal_training = self.config.get("task.reciprocal_training")
        # self.filter_method = self.config.get("data.filter")

        self.train_raw = []
        self.valid_raw = []
        self.test_raw = []

        self.ent2id = defaultdict(None)
        self.rel2id = defaultdict(None)
        self.ts2id = defaultdict(None)

        self.train_set = defaultdict(list)
        self.valid_set = defaultdict(list)
        self.test_set = defaultdict(list)

        self.all_triples = []
        self.all_quadruples = []

        self.load()
        self.process()
        self.filter()

    @classmethod
    def create(cls, config: Config):
        """Factory method for data creation"""

        ds_type: str = config.get("dataset.name")

        if ds_type in DatasetProcessor.list_available():
            kwargs = config.get("dataset.args")  # TODO: 需要改成key的格式
            return DatasetProcessor.by_name(ds_type)(config)  # return an instance
        else:
            raise ConfigurationError(
                f"{ds_type} specified in configuration file is not supported"
                f"implement your data class with `DatasetProcessor.register(name)"
            )

    @abstractmethod
    def process(self):
        raise NotImplementedError

    def index_entities(self, ent: str):
        if ent not in self.ent2id:
            self.ent2id.update({ent: self.num_entities()})

        return self.ent2id[ent]

    def index_relations(self, rel: str):
        if rel not in self.rel2id:
            self.rel2id.update({rel: self.num_relations()})

        return self.rel2id[rel]

    def index_timestamps(self, ts):
        if ts not in self.ts2id:
            self.ts2id.update({ts: self.num_timestamps()})

        return self.ts2id[ts]

    def create_year2id(self, triple_time):
        year2id = dict()
        freq = defaultdict(int)
        year_list = []
        for k, v in triple_time.items():  # v:[start_year:str,end_year:str]
            try:
                start = v[0].split('-')[0]
                end = v[1].split('-')[0]
            except:
                pdb.set_trace()  # 程序暂停

            if start.find('#') == -1 and len(start) == 4:
                year_list.append(start)
            if end.find('#') == -1 and len(end) == 4:
                year_list.append(end)
        # 如果start_year和end_year里没有’#‘且为4位，则存入year_list：[start_year,end_year:str]
        year_list.sort()  # 按年份升序排列
        for year in year_list:  # 统计每个年份出现的频率
            freq[year] = freq[year] + 1
        year_class = []
        count = 0
        for key in sorted(freq.keys()):  # 继续升序提取出年份
            count += freq[key]
            if count > 300:
                year_class.append(key)  # 如果出现频率大于300次（start或者end都可以）
                count = 0
        prev_year = '0'
        i = 0
        for i, yr in enumerate(year_class):
            year2id[(prev_year, yr)] = i  # { (0,yr[0]):0,(yr[0]+1,yr[1]):1,...,(yr[-1]+1,max_year):N}
            prev_year = str(int(yr) + 1)
        year2id[(prev_year, max(year_list))] = i + 1

        return year2id

    def create_id_labels(self, triple_time):
        YEARMAX = '3000'
        YEARMIN = '-50'

        inp_idx, start_idx, end_idx = [], [], []
        for k, v in triple_time.items():
            start = v[0].split('-')[0]
            end = v[1].split('-')[0]
            if start == '####':
                start = YEARMIN
            elif start.find('#') != -1 or len(start) != 4:  # 如果字符串内包含#或者年份的长度不等于4，跳出本次循环
                continue
            if end == '####':
                end = YEARMAX
            elif end.find('#') != -1 or len(end) != 4:
                continue
            if start > end:
                end = YEARMAX

            inp_idx.append(k)
            if start == YEARMIN:
                start_idx.append(0)
            else:
                for key, lbl in sorted(self.year2id.items(), key=lambda x: x[1]):
                    if start >= key[0] and start <= key[1]:
                        start_idx.append(lbl)
            if end == YEARMAX:
                end_idx.append(len(self.year2id.keys()) - 1)
            else:
                for key, lbl in sorted(self.year2id.items(), key=lambda x: x[1]):
                    if end >= key[0] and end <= key[1]:
                        end_idx.append(lbl)

        return inp_idx, start_idx, end_idx

    def get_pretreated_data(self, triple_time, triples):
        inp_idx, start_idx, end_idx = self.create_id_labels(triple_time)
        keep_idx = set(inp_idx)
        for i in range(len(triples) - 1, -1, -1):
            if i not in keep_idx:
                del triples[i]

        posh, rela, post = zip(*triples)
        head, rel, tail = zip(*triples)
        posh = list(posh)
        post = list(post)
        rela = list(rela)
        head = list(head)
        tail = list(tail)
        rel = list(rel)
        for i in range(len(posh)):
            if start_idx[i] < end_idx[i]:
                for j in range(start_idx[i] + 1, end_idx[i] + 1):
                    head.append(posh[i])
                    rel.append(rela[i])
                    tail.append(post[i])
                    start_idx.append(j)
        pretreated_data = []
        for i in range(len(head)):
            pretreated_data.append([head[i], rel[i], tail[i], start_idx[i]])

        return pretreated_data

    def load(self):
        train_file = self.folder + "/train.txt"
        valid_file = self.folder + "/valid.txt"
        test_file = self.folder + "/test.txt"

        if self.name == 'wiki' or self.name == 'yago11k':
            train_triples, valid_triples, test_triples = [], [], []
            train_triple_time, valid_triple_time, test_triple_time = dict(), dict(), dict()
            with open(train_file, 'r') as filein:
                count = 0
                for line in filein:
                    train_triples.append([x.strip() for x in line.split()[0:3]])
                    train_triple_time[count] = [x.split('-')[0] for x in line.split()[3:5]]
                    count += 1
            self.year2id = self.create_year2id(train_triple_time)
            self.train_raw = self.get_pretreated_data(train_triple_time,train_triples)
            self.train_size = len(self.train_raw)

            with open(valid_file, 'r') as filein:
                count = 0
                for line in filein:
                    valid_triples.append([x.strip() for x in line.split()[0:3]])
                    valid_triple_time[count] = [x.split('-')[0] for x in line.split()[3:5]]
                    count += 1
            self.valid_raw = self.get_pretreated_data(valid_triple_time,valid_triples)
            self.valid_size = len(self.valid_raw)

            with open(test_file, 'r') as filein:
                count = 0
                for line in filein:
                    test_triples.append([x.strip() for x in line.split()[0:3]])
                    test_triple_time[count] = [x.split('-')[0] for x in line.split()[3:5]]
                    count += 1
            self.test_raw = self.get_pretreated_data(test_triple_time,test_triples)
            self.test_size = len(self.test_raw)
            self.max_year = len(self.year2id)

        else:
            with open(train_file, "r") as f:
                if self.reciprocal_training:
                    for line in f.readlines():
                        self.train_raw.append(line)

                        insert_line = line.strip().split('\t')
                        insert_line[1] += '(RECIPROCAL)'
                        insert_line[0], insert_line[2] = insert_line[2], insert_line[0]
                        insert_line = '\t'.join(insert_line)

                        self.train_raw.append(insert_line)
                else:
                    self.train_raw = f.readlines()

                self.train_size = len(self.train_raw)

            with open(valid_file, "r") as f:
                self.valid_raw = f.readlines()

                self.valid_size = len(self.valid_raw)

            with open(test_file, "r") as f:
                self.test_raw = f.readlines()

                self.test_size = len(self.test_raw)

    @abstractmethod
    def process_time(self, origin: str):
        # TODO(gengyuan) use datetime
        raise NotImplementedError

    def get(self, split: str = "train"):
        # TODO(gengyuan)
        return {"train": self.train_set, "valid": self.valid_set, "test": self.test_set}[split]

    def num_entities(self):
        return len(self.ent2id)

    def num_relations(self):
        return len(self.rel2id)

    def num_timestamps(self):
        if self.name == 'wiki' or self.name == 'yago11k':
            return self.max_year
        return len(self.ts2id)

    def num_time_identifier(self):
        if self.name == 'wiki' or self.name == 'yago11k':
            return self.max_year
        return len(self.ts2id)

    def filter(self, type="static", target="o") -> Dict[str, List]:
        """
        Returns generated link prediction queries.
        Removes the specified target (either s, p or o) out of a copy of each triple respectively quadruple
        (if specified type is static respectively time-aware) and adds each answer as the last element.
        """
        self.config.assert_true(type in ["static",
                                         "time-aware",
                                         "off"],
                                f"{type} filtering is not implemented; use static/time-aware/off filtering.")
        self.config.assert_true(target in ["s", "p", "o"],
                                "Only support s(ubject)/p(redicate)/o(bject) prediction task")

        filtered_data = defaultdict(list)

        if type != "off":
            all_tuples = self.all_triples if type == "static" else self.all_quadruples

            for tup in all_tuples:
                query = tup.copy()

                # TODO(gengyuan) enum
                missing = query[SPOT[target].value - 1]
                query[SPOT[target].value - 1] = None

                query_k = f"{query[0]}-{query[1]}-{query[2]}"

                if type == "time-aware":
                    query_k += f"-{query[3]}"

                filtered_data[query_k].append(missing)

        return filtered_data

    def info(self):
        self.config.log('==============================================')
        self.config.log(f'Dataset type : {self.config.get("dataset.name")}')
        self.config.log(f"Number of entities : {self.num_entities()}")
        self.config.log(f"Number of relations : {self.num_relations()}")
        self.config.log(f"Number of temporal identifiers : {self.num_timestamps()}")
        self.config.log(f"\n")
        self.config.log(f"Train set size : {self.train_size}")
        self.config.log(f"Valid set size : {self.valid_size}")
        self.config.log(f"Test set size : {self.test_size}")
        self.config.log('==============================================')


@DatasetProcessor.register(name="gdelt")
class GDELTDatasetProcessor(DatasetProcessor):
    def __init__(self, config: Config):
        super().__init__(config)

    def process(self):
        for rd in self.train_raw:
            head, rel, tail, ts = rd.strip().split('\t')
            head = int(head)
            rel = int(rel)
            tail = int(tail)
            ts = self.process_time(ts)
            ts_id = self.index_timestamps(ts)

            self.train_set['triple'].append([head, rel, tail])
            self.train_set['timestamp_id'].append([ts_id])
            self.train_set['timestamp_float'].append(list(map(lambda x: int(x), ts.split('-'))))

            self.all_triples.append([head, rel, tail])
            self.all_quadruples.append([head, rel, tail, ts_id])

        for rd in self.valid_raw:
            head, rel, tail, ts = rd.strip().split('\t')
            head = int(head)
            rel = int(rel)
            tail = int(tail)
            ts = self.process_time(ts)
            ts_id = self.index_timestamps(ts)

            self.valid_set['triple'].append([head, rel, tail])
            self.valid_set['timestamp_id'].append([ts_id])
            self.valid_set['timestamp_float'].append(list(map(lambda x: int(x), ts.split('-'))))

            self.all_triples.append([head, rel, tail])
            self.all_quadruples.append([head, rel, tail, ts_id])

        for rd in self.test_raw:
            head, rel, tail, ts = rd.strip().split('\t')
            head = int(head)
            rel = int(rel)
            tail = int(tail)
            ts = self.process_time(ts)
            ts_id = self.index_timestamps(ts)

            self.test_set['triple'].append([head, rel, tail])
            self.test_set['timestamp_id'].append([ts_id])
            self.test_set['timestamp_float'].append(list(map(lambda x: int(x), ts.split('-'))))

            self.all_triples.append([head, rel, tail])
            self.all_quadruples.append([head, rel, tail, ts_id])

    def process_time(self, origin: str, resolution: str = 'day'):
        all_resolutions = ['year', 'month', 'day', 'hour', 'minute', 'second']
        self.config.assert_true(resolution in all_resolutions, f"Time granularity should be {all_resolutions}")

        ts = origin.split('-') + ['00', '00', '00']
        ts = ts[:all_resolutions.index(resolution) + 1]
        ts = '-'.join(ts)

        return ts


@DatasetProcessor.register(name="icews14")
class ICEWS14DatasetProcessor(DatasetProcessor):
    def process(self):
        all_timestamp = get_all_days_of_year(2014)
        self.ts2id = {ts: (arrow.get(ts) - arrow.get('2014-01-01')).days for ts in all_timestamp}

        for rd in self.train_raw:
            head, rel, tail, ts = rd.strip().split('\t')
            head = self.index_entities(head)
            rel = self.index_relations(rel)
            tail = self.index_entities(tail)
            ts = self.process_time(ts)
            ts_id = self.index_timestamps(ts)

            self.train_set['triple'].append([head, rel, tail])
            self.train_set['timestamp_id'].append([ts_id])
            self.train_set['timestamp_float'].append(list(map(lambda x: int(x), ts.split('-'))))

            self.all_triples.append([head, rel, tail])
            self.all_quadruples.append([head, rel, tail, ts_id])

        for rd in self.valid_raw:
            head, rel, tail, ts = rd.strip().split('\t')
            head = self.index_entities(head)
            rel = self.index_relations(rel)
            tail = self.index_entities(tail)
            ts = self.process_time(ts)
            ts_id = self.index_timestamps(ts)

            self.valid_set['triple'].append([head, rel, tail])
            self.valid_set['timestamp_id'].append([ts_id])
            self.valid_set['timestamp_float'].append(list(map(lambda x: int(x), ts.split('-'))))

            self.all_triples.append([head, rel, tail])
            self.all_quadruples.append([head, rel, tail, ts_id])

        for rd in self.test_raw:
            head, rel, tail, ts = rd.strip().split('\t')
            head = self.index_entities(head)
            rel = self.index_relations(rel)
            tail = self.index_entities(tail)
            ts = self.process_time(ts)
            ts_id = self.index_timestamps(ts)

            self.test_set['triple'].append([head, rel, tail])
            self.test_set['timestamp_id'].append([ts_id])
            self.test_set['timestamp_float'].append(list(map(lambda x: int(x), ts.split('-'))))

            self.all_triples.append([head, rel, tail])
            self.all_quadruples.append([head, rel, tail, ts_id])

    def process_time(self, origin: str):
        all_resolutions = ['year', 'month', 'day', 'hour', 'minute', 'second']
        self.config.assert_true(self.resolution in all_resolutions, f"Time granularity should be {all_resolutions}")

        ts = origin.split('-') + ['00', '00', '00']
        ts = ts[:all_resolutions.index(self.resolution) + 1]
        ts = '-'.join(ts)

        return ts


@DatasetProcessor.register(name="icews11-14")
class ICEWS1114DatasetProcessor(DatasetProcessor):
    def process(self):
        all_timestamp = get_all_days_of_year(2011) + \
                        get_all_days_of_year(2012) + \
                        get_all_days_of_year(2013) + \
                        get_all_days_of_year(2014)
        self.ts2id = {ts: (arrow.get(ts) - arrow.get('2011-01-01')).days for ts in all_timestamp}

        for rd in self.train_raw:
            head, rel, tail, ts = rd.strip().split('\t')
            head = self.index_entities(head)
            rel = self.index_relations(rel)
            tail = self.index_entities(tail)
            ts = self.process_time(ts)
            ts_id = self.index_timestamps(ts)

            self.train_set['triple'].append([head, rel, tail])
            self.train_set['timestamp_id'].append([ts_id])
            self.train_set['timestamp_float'].append(list(map(lambda x: int(x), ts.split('-'))))

            self.all_triples.append([head, rel, tail])
            self.all_quadruples.append([head, rel, tail, ts_id])

        for rd in self.valid_raw:
            head, rel, tail, ts = rd.strip().split('\t')
            head = self.index_entities(head)
            rel = self.index_relations(rel)
            tail = self.index_entities(tail)
            ts = self.process_time(ts)
            ts_id = self.index_timestamps(ts)

            self.valid_set['triple'].append([head, rel, tail])
            self.valid_set['timestamp_id'].append([ts_id])
            self.valid_set['timestamp_float'].append(list(map(lambda x: int(x), ts.split('-'))))

            self.all_triples.append([head, rel, tail])
            self.all_quadruples.append([head, rel, tail, ts_id])

        for rd in self.test_raw:
            head, rel, tail, ts = rd.strip().split('\t')
            head = self.index_entities(head)
            rel = self.index_relations(rel)
            tail = self.index_entities(tail)
            ts = self.process_time(ts)
            ts_id = self.index_timestamps(ts)

            self.test_set['triple'].append([head, rel, tail])
            self.test_set['timestamp_id'].append([ts_id])
            self.test_set['timestamp_float'].append(list(map(lambda x: int(x), ts.split('-'))))

            self.all_triples.append([head, rel, tail])
            self.all_quadruples.append([head, rel, tail, ts_id])

    def process_time(self, origin: str):
        all_resolutions = ['year', 'month', 'day', 'hour', 'minute', 'second']
        self.config.assert_true(self.resolution in all_resolutions, f"Time granularity should be {all_resolutions}")

        ts = origin.split('-') + ['00', '00', '00']
        ts = ts[:all_resolutions.index(self.resolution) + 1]
        ts = '-'.join(ts)

        return ts


@DatasetProcessor.register(name="icews05-15")
class ICEWS0515DatasetProcessor(DatasetProcessor):
    def process(self):
        for rd in self.train_raw:
            head, rel, tail, ts = rd.strip().split('\t')
            head = self.index_entities(head)
            rel = self.index_relations(rel)
            tail = self.index_entities(tail)
            ts = self.process_time(ts)

            self.train_set['triple'].append([head, rel, tail])
            self.train_set['timestamp_id'].append([self.index_timestamps(ts)])
            self.train_set['timestamp_float'].append(list(map(lambda x: int(x), ts.split('-'))))

        for rd in self.train_raw:
            head, rel, tail, ts = rd.strip().split('\t')
            head = self.index_entities(head)
            rel = self.index_relations(rel)
            tail = self.index_entities(tail)
            ts = self.process_time(ts)

            self.valid_set['triple'].append([head, rel, tail])
            self.valid_set['timestamp_id'].append([self.index_timestamps(ts)])
            self.valid_set['timestamp_float'].append(list(map(lambda x: int(x), ts.split('-'))))

        for rd in self.train_raw:
            head, rel, tail, ts = rd.strip().split('\t')
            head = self.index_entities(head)
            rel = self.index_relations(rel)
            tail = self.index_entities(tail)
            ts = self.process_time(ts)

            self.test_set['triple'].append([head, rel, tail])
            self.test_set['timestamp_id'].append([self.index_timestamps(ts)])
            self.test_set['timestamp_float'].append(list(map(lambda x: int(x), ts.split('-'))))

    def process_time(self, origin: str):
        raise NotImplementedError


@DatasetProcessor.register(name="wiki")
class WIKIDatasetProcessor(DatasetProcessor):
    def __init__(self, config: Config):
        super().__init__(config)

    def process(self):
        for rd in self.train_raw:
            head, rel, tail, ts = rd[0], rd[1], rd[2], rd[3]
            head = self.index_entities(head)
            rel = self.index_relations(rel)
            tail = self.index_entities(tail)
            ts = int(ts)

            self.train_set['triple'].append([head, rel, tail])
            self.train_set['timestamp_id'].append([ts])
            # self.train_set['timestamp_float'].append(list(map(lambda x: int(x), ts.split('-'))))

            self.all_triples.append([head, rel, tail])
            self.all_quadruples.append([head, rel, tail, ts])

        for rd in self.valid_raw:
            head, rel, tail, ts = rd[0], rd[1], rd[2], rd[3]
            head = self.index_entities(head)
            rel = self.index_relations(rel)
            tail = self.index_entities(tail)
            ts = int(ts)

            self.valid_set['triple'].append([head, rel, tail])
            self.valid_set['timestamp_id'].append([ts])
            # self.valid_set['timestamp_float'].append(list(map(lambda x: int(x), ts.split('-'))))

            self.all_triples.append([head, rel, tail])
            self.all_quadruples.append([head, rel, tail, ts])

        for rd in self.test_raw:
            head, rel, tail, ts = rd[0], rd[1], rd[2], rd[3]
            head = self.index_entities(head)
            rel = self.index_relations(rel)
            tail = self.index_entities(tail)
            ts = int(ts)

            self.test_set['triple'].append([head, rel, tail])
            self.test_set['timestamp_id'].append([ts])
            # self.test_set['timestamp_float'].append(list(map(lambda x: int(x), ts.split('-'))))

            self.all_triples.append([head, rel, tail])
            self.all_quadruples.append([head, rel, tail, ts])

    def process_time(self, origin: str, resolution: str = 'year'):
        pass


@DatasetProcessor.register(name="yago11k")
class YAGODatasetProcessor(DatasetProcessor):
    def __init__(self, config: Config):
        super().__init__(config)

    def process(self):
        for rd in self.train_raw:
            head, rel, tail, ts = rd[0], rd[1], rd[2], rd[3]
            head = self.index_entities(head)
            rel = self.index_relations(rel)
            tail = self.index_entities(tail)
            ts = int(ts)

            self.train_set['triple'].append([head, rel, tail])
            self.train_set['timestamp_id'].append([ts])
            # self.train_set['timestamp_float'].append(list(map(lambda x: int(x), ts.split('-'))))

            self.all_triples.append([head, rel, tail])
            self.all_quadruples.append([head, rel, tail, ts])

        for rd in self.valid_raw:
            head, rel, tail, ts = rd[0], rd[1], rd[2], rd[3]
            head = self.index_entities(head)
            rel = self.index_relations(rel)
            tail = self.index_entities(tail)
            ts = int(ts)

            self.valid_set['triple'].append([head, rel, tail])
            self.valid_set['timestamp_id'].append([ts])
            # self.valid_set['timestamp_float'].append(list(map(lambda x: int(x), ts.split('-'))))

            self.all_triples.append([head, rel, tail])
            self.all_quadruples.append([head, rel, tail, ts])

        for rd in self.test_raw:
            head, rel, tail, ts = rd[0], rd[1], rd[2], rd[3]
            head = self.index_entities(head)
            rel = self.index_relations(rel)
            tail = self.index_entities(tail)
            ts = int(ts)

            self.test_set['triple'].append([head, rel, tail])
            self.test_set['timestamp_id'].append([ts])
            # self.test_set['timestamp_float'].append(list(map(lambda x: int(x), ts.split('-'))))

            self.all_triples.append([head, rel, tail])
            self.all_quadruples.append([head, rel, tail, ts])

    def process_time(self, origin: str, resolution: str = 'year'):
        pass


class SplitDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: Dict[str, List], datatype: Optional[List[str]] = None):
        super().__init__()

        self.dataset = dataset
        self.datatype: List = datatype

        # TODO(gengyuan) assert the lengths of all lists in self.dataset
        # use self.config.assert_true(condition, message)
        # assert all( for i in dataset.items())

    def __len__(self):
        # TODO(gengyuan) calculate the length
        return len(self.dataset['triple'])

    def __getitem__(self, index, train=True):
        sample = torch.Tensor(self.dataset['triple'][index])

        for type in self.datatype:
            if type == 'timestamp_id':
                timestamp_id = torch.Tensor(self.dataset['timestamp_id'][index])
                sample = torch.cat([sample, timestamp_id], dim=0)

            elif type == 'timestamp_float':
                timestamp_float = torch.Tensor(self.dataset['timestamp_float'][index])
                sample = torch.cat([sample, timestamp_float], dim=0)
            else:
                raise NotImplementedError

        return sample
