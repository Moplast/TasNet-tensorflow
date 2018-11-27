'''
Required:
mixture: [B, K, L]
source: [B, nspk, K, L]
get_batch
'''

import numpy as np
import librosa
import os
from params import *
import pickle


class DataGenerator(object):
    def __init__(self, batch_size, max_k,
                 save_dir=None, data_dir=None, name='data_gen'):
        self.name = name
        self.batch_size = batch_size
        self.max_k = max_k
        self.data_dir = data_dir
        self.save_dir = save_dir
        if save_dir is not None and not os.path.exists(save_dir):
            os.mkdir(save_dir)
        self.data_subdir = ['s1', 's2', 'mix']
        self.data_type = ['tr', 'cv']

        self.spks = []
        self.init_samples()

        self.epoch = 0
        self.idx = 0

    def init_samples(self):
        self.samples = {'mix': [], 's': []}
        self.sample_size = 0

    def gen_data(self):
        if self.data_dir and self.save_dir is None:
            raise AssertionError
        for dt in self.data_type:
            self.init_samples()
            save_cnt = 1
            save_dir = os.path.join(self.save_dir, dt)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            dt_path = os.path.join(self.data_dir, dt)
            dt_mix_path = os.path.join(dt_path, 'mix')
            dt_s1_path = os.path.join(dt_path, 's1')
            dt_s2_path = os.path.join(dt_path, 's2')

            list_mix = os.listdir(dt_mix_path)
            for wav_file in list_mix:
                if not wav_file.endswith('.wav'):
                    continue
                # print(wav_file)
                mix_path = os.path.join(dt_mix_path, wav_file)
                s1_path = os.path.join(dt_s1_path, wav_file)
                s2_path = os.path.join(dt_s2_path, wav_file)

                spk1 = wav_file[:3]
                spk2 = wav_file.split(sep='_')[2][:3]

                if spk1 not in self.spks:
                    self.spks.append(spk1)
                if spk2 not in self.spks:
                    self.spks.append(spk2)

                mix, _ = librosa.load(mix_path, sr=sr)
                s1, _ = librosa.load(s1_path, sr=sr)
                s2, _ = librosa.load(s2_path, sr=sr)
                self.get_sample(mix, s1, s2, [spk1, spk2])

                self.sample_size = len(self.samples['mix'])
                if self.sample_size % 50 == 0:
                    print(self.sample_size)
                if self.sample_size >= save_cnt * 50000:
                    pickle.dump(self.samples,
                                open(save_dir + '/raw_' + str(self.max_k) + '-' + str(self.sample_size) + '.pkl',
                                     'wb'))
                    save_cnt += 1
            pickle.dump(self.samples,
                        open(save_dir + '/raw_' + str(self.max_k) + '-' + str(self.sample_size) + '.pkl',
                             'wb'))

    def get_sample(self, mix, s1, s2, spks):
        spk_num = len(spks)

        mix_len = len(mix)
        sample_num = int(np.ceil(mix_len / L))
        if sample_num < self.max_k:
            sample_num = self.max_k
        max_len = sample_num * L
        pad_s1 = np.concatenate([s1, np.zeros([max_len - len(s1)])])
        pad_s2 = np.concatenate([s2, np.zeros([max_len - len(s1)])])
        pad_mix = np.concatenate([mix, np.zeros([max_len - len(mix)])])

        k_ = 0
        while k_ + self.max_k <= sample_num:
            begin = k_ * L
            end = (k_ +self.max_k) * L
            sample_mix = pad_mix[begin:end]
            sample_s1 = pad_s1[begin:end]
            sample_s2 = pad_s2[begin:end]

            sample_mix = np.reshape(sample_mix, [self.max_k, L])
            sample_s1 = np.reshape(sample_s1, [self.max_k, L])
            sample_s2 = np.reshape(sample_s2, [self.max_k, L])
            sample_s = np.dstack((sample_s1, sample_s2))
            sample_s = np.transpose(sample_s, (2, 0, 1))

            self.samples['mix'].append(sample_mix)
            self.samples['s'].append(sample_s)
            k_ += self.max_k

    def load_data(self, data_path):
        self.samples = pickle.load(open(data_path, 'rb'))
        self.sample_size = len(self.samples['mix'])
        print('>> {0}: Loading samples from pkl: {1}...'.format(self.name, data_path))

    def shuffle_dict(self):
        rand_per = np.random.permutation(self.sample_size)
        self.samples['mix'] = np.array(self.samples['mix'])[rand_per]
        self.samples['s'] = np.array(self.samples['s'])[rand_per]

    def get_a_sample(self, mix, s1, s2, spks, max_k):
        spk_num = len(spks)
        mix_len = len(mix)
        sample_num = int(np.ceil(mix_len / L / max_k)) * max_k
        max_len = sample_num * L
        pad_s1 = np.concatenate([s1, np.zeros([max_len - len(s1)])])
        pad_s2 = np.concatenate([s2, np.zeros([max_len - len(s1)])])
        pad_mix = np.concatenate([mix, np.zeros([max_len - len(mix)])])

        test_sample = {
            'mix': [],
            's': [],
        }
        k_ = 0
        while k_ + self.max_k <= sample_num:
            begin = k_ * L
            end = (k_ + max_k) * L
            sample_mix = pad_mix[begin:end]
            sample_s1 = pad_s1[begin:end]
            sample_s2 = pad_s2[begin:end]

            sample_mix = np.reshape(sample_mix, [max_k, L])
            sample_s1 = np.reshape(sample_s1, [max_k, L])
            sample_s2 = np.reshape(sample_s2, [max_k, L])
            sample_s = np.dstack((sample_s1, sample_s2))
            sample_s = np.transpose(sample_s, (2, 0, 1))

            test_sample['mix'].append(sample_mix)
            test_sample['s'].append(sample_s)

            k_ += max_k

        return test_sample

    def gen_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        n_begin = self.idx
        n_end = self.idx + batch_size
        if n_end >= self.sample_size:
            # rewire the index
            self.idx = 0
            n_begin = self.idx
            n_end = self.idx + batch_size
            self.epoch += 1
            self.shuffle_dict()
        self.idx += batch_size
        samples = {
            'mix': self.samples['mix'][n_begin: n_end],
            's': self.samples['s'][n_begin: n_end]
        }
        return samples


if __name__ == '__main__':
    # data_gen = DataGenerator(batch_size=1, max_k=int(0.5/0.005), save_dir='/home/grz/data/SSSR/wsj0_tasnet/',
    #                          data_dir='/home/grz/data/SSSR/wsj0/min/',
    #                          name='gen_data')
    # data_gen.gen_data()
    # data_gen = DataGenerator(batch_size=1, max_k=int(4/0.005), save_dir='/home/grz/data/SSSR/wsj0_tasnet/',
    #                          data_dir='/home/grz/data/SSSR/wsj0/min/',
    #                          name='gen_data')
    # data_gen.gen_data()

    data_gen = DataGenerator(batch_size=1, max_k=int(2/0.005), save_dir='/home/grz/data/SSSR/wsj0_tasnet/',
                             data_dir='/home/grz/data/SSSR/wsj0/min/',
                             name='gen_data')
    data_gen.gen_data()