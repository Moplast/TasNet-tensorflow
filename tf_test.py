import tensorflow as tf
import numpy as np
from tasnet.params import *
from tasnet.tf_net import TasNet
from tasnet.mir_eval import bss_eval_sources
from tasnet.data_generator import DataGenerator
import os
from datetime import datetime
import time
import librosa
import matlab.engine
import pickle

from itertools import product, permutations

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
seq_duration = 0.5
seq_len = int(seq_duration / sample_duration)

model_subpath = '100-e19.ckpt-32641'
model_path = os.getcwd() + '/' + sum_dir + '/model/' + model_subpath


def evaluate(data_type):
    wav_dir = os.path.join(data_dir, data_type)
    spk_gender = pickle.load(open('/home/grz/SS/MESID/dataset/wsj0_spk_gender.pkl', 'rb'))
    mix_dir = os.path.join(wav_dir, 'mix')
    s1_dir = os.path.join(wav_dir, 's1')
    s2_dir = os.path.join(wav_dir, 's2')

    list_mix = os.listdir(mix_dir)
    list_mix.sort()
    if data_type == 'tt':  # 3000
        factor = 50
    else:
        factor = 2
    list_mix = list_mix[::factor]
    np.random.shuffle(list_mix)

    segment_num = 0
    MATLAB = matlab.engine.start_matlab()

    # =============== PLACEHOLDER & MODEL DEFINITION ========================
    with tf.Graph().as_default():
        mixture = tf.placeholder(shape=[None, None, L], dtype=tf.float32, name='mixture')
        source = tf.placeholder(shape=[None, nspk, None, L], dtype=tf.float32, name='source')

        print('>> Initializing model...')
        model = TasNet(batch_size=1, seq_len=seq_len)
        est_source = model.build_network(mixture)
        loss, max_snr, reest_source, reorder_recon = model.objective(est_source=est_source, source=source)

        saver = tf.train.Saver(tf.global_variables())
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        saver.restore(sess, model_path)
        print('load model from: ', model_path)

        data_gen = DataGenerator(batch_size=1, max_k=seq_len, name='eval-generator')

        print('>> Evaluating...  %s start' % datetime.now())

        for wav_file in list_mix:
            gender_mix = 'dg'
            print('# segment: ', segment_num)
            if not wav_file.endswith('.wav'):
                continue

            gender1 = spk_gender[wav_file.split(sep='_')[0][:3]]
            gender2 = spk_gender[wav_file.split(sep='_')[2][:3]]
            if gender1 == gender2:
                print('>> same gender')
                gender_mix = 'sg'
            else:
                print('>> diff gender')

            print("Sentence {0}, gender_mix: {1}".format(wav_file, gender_mix))
            mix_path = os.path.join(mix_dir, wav_file)
            s1_path = os.path.join(s1_dir, wav_file)
            s2_path = os.path.join(s2_dir, wav_file)
            spk1 = wav_file[:3]
            spk2 = wav_file.split(sep='_')[2][:3]

            mix, _ = librosa.load(mix_path, sr=sr)
            s1, _ = librosa.load(s1_path, sr=sr)
            s2, _ = librosa.load(s2_path, sr=sr)
            mix_len = len(mix)
            test_sample = data_gen.get_a_sample(mix, s1, s2, spks=[spk1, spk2], max_k=seq_len)
            sample_num = len(test_sample['mix'])
            # utterance-level info
            est_s_u = []
            snr_u = []

            start_time = datetime.now()
            for i in range(sample_num):
                est_source_np, ordered_est_source_np, max_snr_np = sess.run([est_source, reorder_recon, max_snr],
                                         feed_dict={mixture: [np.array(test_sample['mix'][i]).astype(np.float32)],
                                                    source: [np.array(test_sample['s'][i]).astype(np.float32)]})
                # (ordered) est_source_np [1, nspk, K, L]
                # max_snr_np [1]
                est_s_u.append(ordered_est_source_np[0])
                snr_u.append(max_snr_np[0])

            duration = (datetime.now() - start_time).seconds
            print('>> past time (s):', duration)
            recon_s1_sig, recon_s2_sig = recover_sig(est_s_u)
            recon_s1_sig = recon_s1_sig[:mix_len]
            recon_s2_sig = recon_s2_sig[:mix_len]

            # sdri = measure_wsj0(MATLAB, mix, recon_s1_sig, recon_s2_sig, s1, s2,
            #              mixture_name='tasnet-sdri')
            src_ref = np.stack([s1, s2], axis=0)
            src_est = np.stack([recon_s1_sig, recon_s2_sig], axis=0)
            src_anchor = np.stack([mix, mix], axis=0)
            sdr, sir, sar, popt = bss_eval_sources(src_ref, src_est)
            sdr0, sir0, sar0, popt0 = bss_eval_sources(src_ref, src_anchor)
            snr1 = get_SISNR(s1, recon_s1_sig)
            snr2 = get_SISNR(s2, recon_s2_sig)

            print("snr1: {}, snr2: {}".format(snr1, snr2))
            print("sdr1: {}, sdr2: {}".format(sdr[0]-sdr0[0], sdr[1]-sdr0[0]))

        sess.close()


def recover_sig(sig_list):
    '''sig_list: [n] - [nspk, K, L]'''
    sig_np = np.concatenate([n for n in sig_list], axis=1)  # [n*K, L]
    sig = np.reshape(sig_np, [nspk, -1])
    return sig[0], sig[1]


def get_time_str():
    time = datetime.now()
    timestr = '%02d%02d%02d-%02d%02d%02d' % (time.year, time.month, time.day,
                                             time.hour, time.minute, time.second)
    return timestr


def get_SISNR(ref_sig, out_sig, eps=1e-8):
    assert len(ref_sig) == len(out_sig)
    ref_sig = ref_sig - np.mean(ref_sig)
    out_sig = out_sig - np.mean(out_sig)
    ref_energy = np.sum(ref_sig ** 2) + eps
    proj = np.sum(ref_sig * out_sig) * ref_sig / ref_energy
    noise = out_sig - proj
    ratio = np.sum(proj ** 2) / (np.sum(noise ** 2) + eps)
    sisnr = 10 * np.log(ratio + eps) / np.log(10.0)
    return sisnr


def measure_wsj0(MATLAB, mix_wav, est_speech1, est_speech2, ori_speech1, ori_speech2,
                 mixture_name=''):

    min_len = min(len(mix_wav), np.array(est_speech1).shape[0], np.array(est_speech2).shape[0],
                  np.array(ori_speech1).shape[0], len(ori_speech2))
    mix_wav = mix_wav[:min_len]
    est_speech2 = est_speech2[:min_len]
    est_speech1 = est_speech1[:min_len]
    ori_speech1 = ori_speech1[:min_len]
    ori_speech2 = ori_speech2[:min_len]

    mix_wav = matlab.double(mix_wav.tolist())
    ori_speech1 = matlab.double(ori_speech1.tolist())
    ori_speech2 = matlab.double(ori_speech2.tolist())
    est_speech1 = matlab.double(est_speech1.tolist())
    est_speech2 = matlab.double(est_speech2.tolist())
    # BSS_EVAL (true_signal, true_noise, pred_signal, mix)

    bss_eval_results11 = MATLAB.BSS_EVAL(ori_speech1, ori_speech2, est_speech1, mix_wav)  # ori_speech
    bss_eval_results21 = MATLAB.BSS_EVAL(ori_speech2, ori_speech1, est_speech2, mix_wav)

    writeline = mixture_name + '\n- speech_1\tSDR: %.2f dB\tSIR: %.2f dB\tSAR: %.2f dB\tNSDR: %.2f dB\n' \
                % (bss_eval_results11['SDR'], bss_eval_results11['SIR'], bss_eval_results11['SAR'],
                   bss_eval_results11['NSDR']) \
                + '- speech_2\tSDR: %.2f dB\tSIR: %.2f dB\tSAR: %.2f dB\tNSDR: %.2f dB' \
                % (bss_eval_results21['SDR'], bss_eval_results21['SIR'], bss_eval_results21['SAR'],
                   bss_eval_results21['NSDR'])
    print(writeline)

    nsdr1 = bss_eval_results11['NSDR']
    if nsdr1 < 0:
        nsdr1 = 0
    nsdr2 = bss_eval_results21['NSDR']
    if nsdr2 < 0:
        nsdr2 = 0
    # for debug
    # if nsdr1 < 4 or nsdr2 < 4:
    #     return None
    # else:
    return nsdr1 + nsdr2

if __name__ == '__main__':
    evaluate('cv')
