import os
import random
import argparse

from functions import Storage
root_dataset_dir = '/usr/zhoujie/myjupyter/MMSA/data/dataset/multimodal-sentiment-dataset/StandardDatasets'

class ConfigTune():
    def __init__(self, args):
        # global parameters for running
        self.globalArgs = args
        # hyper parameters for models
        HYPER_MODEL_MAP = {
            'mifcu': self.__MIF,
        }
        # hyper parameters for datasets
        HYPER_DATASET_MAP = self.__datasetCommonParams()
        # normalize
        model_name = str.lower(args.modelName)
        dataset_name = str.lower(args.datasetName)
        # load params
        commonArgs = HYPER_MODEL_MAP[model_name]()['commonParas']
        dataArgs = HYPER_DATASET_MAP[dataset_name]
        dataArgs = dataArgs['aligned'] if (commonArgs['need_data_aligned'] and 'aligned' in dataArgs) else dataArgs['unaligned']
        # integrate all parameters
        self.args = Storage(dict(vars(args),
                            **dataArgs,
                            **commonArgs,
                            **HYPER_MODEL_MAP[model_name]()['debugParas'],
                            ))
    
    def __datasetCommonParams(self):
        tmp = {
            'mosi':{
                'aligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSI/Processed/aligned_50.pkl'),
                    'seq_lens': (50, 50, 50),
                    # (text, audio, video)
                    'feature_dims': (768, 5, 20),
                    'train_samples': 1284,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss',
                    'H': 3.0
                },
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSI/Processed/unaligned_50.pkl'),
                    'seq_lens': (50, 50, 50),
                    # (text, audio, video)
                    'feature_dims': (768, 5, 20),
                    'train_samples': 1284,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss',
                    'H': 3.0
                }
            },
            'mosei':{
                'aligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSEI/Processed/aligned_50.pkl'),
                    'seq_lens': (50, 50, 50),
                    # (text, audio, video)
                    'feature_dims': (768, 74, 35),
                    'train_samples': 16326,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss',
                    'H': 3.0
                },
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSEI/Processed/unaligned_50.pkl'),
                    'seq_lens': (50, 500, 375),
                    # (text, audio, video)
                    'feature_dims': (768, 74, 35),
                    'train_samples': 16326,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss',
                    'H': 3.0
                }
            },
            'sims':{
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'SIMS/Processed/features/unaligned_39.pkl'),
                    # (batch_size, seq_lens, feature_dim)
                    'seq_lens': (39, 400, 55), # (text, audio, video)
                    'feature_dims': (768, 33, 709), # (text, audio, video)
                    'train_samples': 1368,
                    'num_classes': 5,
                    'language': 'cn',
                    'KeyEval': 'Loss',
                    'H': 1.0
                }
            }
        }
        return tmp

    def __MIF(self):
        tmp = {
            'commonParas':{
                'need_data_aligned': False,
                'need_model_aligned': False,
                'use_finetune': True,
                'use_bert': True,
                'early_stop': 12,
                'update_epochs': 2,
                'rnncell': 'lstm',
                'use_cmd_sim': True,
            },
            'debugParas':{
                'd_paras': ['batch_size', 'learning_rate', 'hidden_size', 'dropout',  'attention_dropout', 'reverse_grad_weight', 'diff_weight', 'sim_weight', \
                'sp_weight', 'grad_clip', 'weight_decay', 'num_head','Decoder_head'],
                'batch_size': random.choice([8,16,32,64]),
                'learning_rate': random.choice([5e-4,1e-5,1e-6,5e-5]),
                'hidden_size': random.choice([32,64,128,256]),
                'dropout': random.choice([0.5, 0.2, 0.0]),
                'attention_dropout': random.choice([0.5, 0.2, 0.0]),
                'reverse_grad_weight': random.choice([0.5, 0.8, 1.0]),
                'diff_weight': random.choice([0.1, 0.3, 0.5]),
                'sim_weight': random.choice([0.5, 0.8, 1.0]),
                'sp_weight': random.choice([0.0, 1.0]),
                # when grad_clip == -1.0, means not use that
                'grad_clip': random.choice([-1.0, 0.8,0.5,1.0]),
                'weight_decay': random.choice([0,1e-3,2e-4,1e-5]),
                'num_head': random.choice([2, 4, 8]),
                'Decoder_head': random.choice([2, 4, 8]),
                }
        }
        return tmp

    def get_config(self):
        return self.args

class ConfigClassification():
    def __init__(self, args):
        # hyper parameters for models
        HYPER_MODEL_MAP = {
            'mif': self.__MIF,
        }
        # hyper parameters for datasets
        HYPER_DATASET_MAP = self.__datasetCommonParams()

        # normalize
        model_name = str.lower(args.modelName)
        dataset_name = str.lower(args.datasetName)
        # load params
        commonArgs = HYPER_MODEL_MAP[model_name]()['commonParas']
        dataArgs = HYPER_DATASET_MAP[dataset_name]
        dataArgs = dataArgs['aligned'] if (commonArgs['need_data_aligned'] and 'aligned' in dataArgs) else dataArgs['unaligned']
        # integrate all parameters
        self.args = Storage(dict(vars(args),
                            **dataArgs,
                            **commonArgs,
                            **HYPER_MODEL_MAP[model_name]()['datasetParas'][dataset_name],
                            ))
    
    def __datasetCommonParams(self):
        # root_dataset_dir = './data/dataset/multimodal-sentiment-dataset/StandardDatasets'
        tmp = {
            'mosi':{
                'aligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSI/Processed/aligned_50.pkl'),
                    'seq_lens': (50, 50, 50),
                    # (text, audio, video)
                    'feature_dims': (768, 5, 20),
                    'train_samples': 1284,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss' 
                },
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSI/Processed/unaligned_50.pkl'),
                    'seq_lens': (50, 50, 50),
                    # (text, audio, video)
                    'feature_dims': (768, 5, 20),
                    'train_samples': 1284,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss' 
                }
            },
            'mosei':{
                'aligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSEI/Processed/aligned_50.pkl'),
                    'seq_lens': (50, 50, 50),
                    # (text, audio, video)
                    'feature_dims': (768, 74, 35),
                    'train_samples': 16326,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss'
                },
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSEI/Processed/unaligned_50.pkl'),
                    'seq_lens': (50, 500, 375),
                    # (text, audio, video)
                    'feature_dims': (768, 74, 35),
                    'train_samples': 16326,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss'
                }
            },
            'sims':{
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'SIMS/Processed/features/unaligned_39.pkl'),
                    # (batch_size, seq_lens, feature_dim)
                    'seq_lens': (39, 400, 55), # (text, audio, video)
                    'feature_dims': (768, 33, 709), # (text, audio, video)
                    'train_samples': 1368,
                    'num_classes': 5,
                    'language': 'cn',
                    'KeyEval': 'Loss',
                }
            }
        }
        return tmp
    def __MIF(self):
        tmp = {
            'commonParas':{
                'need_data_aligned': False,
                'need_model_aligned': False,
                'use_finetune': True,
                'use_bert': True,
                'early_stop': 8,
                'update_epochs': 2,
                'rnncell': 'lstm',
                'use_cmd_sim': True,
            },
            # dataset
            'datasetParas':{
                'mosi':{
                    # the batch_size of each epoch is updata_epochs * batch_size
                    'batch_size': 64,
                    'learning_rate': 1e-4,
                    'hidden_size': 128,
                    'dropout': 0.2,
                    'reverse_grad_weight': 0.8,
                    'diff_weight': 0.3,
                    'sim_weight': 0.8,
                    'sp_weight': 0.0,
                    # when grad_clip == -1.0, means not use that
                    'grad_clip': 1.0,
                    'weight_decay': 0.002,
                },
                'mosei':{
                    # the batch_size of each epoch is updata_epochs * batch_size
                    'batch_size': 32,
                    'learning_rate': 1e-4,
                    'hidden_size': 128,
                    'dropout': 0.2,
                    'reverse_grad_weight': 0.5,
                    'diff_weight': 0.1,
                    'sim_weight': 1.0,
                    'sp_weight': 1.0,
                    # when grad_clip == -1.0, means not use that
                    'grad_clip': 0.8,
                    'weight_decay': 0.0,
                },
                'sims':{
                    # the batch_size of each epoch is updata_epochs * batch_size
                    'batch_size': 32,
                    'learning_rate': 5e-5,
                    'hidden_size': 128,
                    'dropout': 0,
                    'attention_dropout':0.2,
                    'reverse_grad_weight': 1,
                    'diff_weight': 0.5,
                    'sim_weight': 0.8,
                    'sp_weight': 1,
                    # when grad_clip == -1.0, means not use that
                    'grad_clip': -1,
                    'weight_decay': 0,
                    'num_head': 4
                }
            },
        }
        return tmp

class ConfigRegression():
    def __init__(self, args):
        # hyper parameters for models
        HYPER_MODEL_MAP = {
            'mif': self.__MIF,
        }
        # hyper parameters for datasets
        HYPER_DATASET_MAP = self.__datasetCommonParams()

        # normalize
        model_name = str.lower(args.modelName)
        dataset_name = str.lower(args.datasetName)
        # load params
        commonArgs = HYPER_MODEL_MAP[model_name]()['commonParas']
        dataArgs = HYPER_DATASET_MAP[dataset_name]
        dataArgs = dataArgs['aligned'] if (commonArgs['need_data_aligned'] and 'aligned' in dataArgs) else dataArgs['unaligned']
        # integrate all parameters
        self.args = Storage(dict(vars(args),
                            **dataArgs,
                            **commonArgs,
                            **HYPER_MODEL_MAP[model_name]()['datasetParas'][dataset_name],
                            ))
    
    def __datasetCommonParams(self):
        # root_dataset_dir = './data/dataset/multimodal-sentiment-dataset/StandardDatasets'
        tmp = {
            'mosi':{
                'aligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSI/Processed/aligned_50.pkl'),
                    'seq_lens': (50, 50, 50),
                    # (text, audio, video)
                    'feature_dims': (768, 5, 20),
                    'train_samples': 1284,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss' 
                },
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSI/Processed/unaligned_50.pkl'),
                    'seq_lens': (50, 50, 50),
                    # (text, audio, video)
                    'feature_dims': (768, 5, 20),
                    'train_samples': 1284,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss' 
                }
            },
            'mosei':{
                'aligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSEI/Processed/aligned_50.pkl'),
                    'seq_lens': (50, 50, 50),
                    # (text, audio, video)
                    'feature_dims': (768, 74, 35),
                    'train_samples': 16326,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss'
                },
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSEI/Processed/unaligned_50.pkl'),
                    'seq_lens': (50, 500, 375),
                    # (text, audio, video)
                    'feature_dims': (768, 74, 35),
                    'train_samples': 16326,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss'
                }
            },
            'sims':{
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'SIMS/Processed/features/unaligned_39.pkl'),
                    # (batch_size, seq_lens, feature_dim)
                    'seq_lens': (39, 400, 55), # (text, audio, video)
                    'feature_dims': (768, 33, 709), # (text, audio, video)
                    'train_samples': 1368,
                    'num_classes': 5,
                    'language': 'cn',
                    'KeyEval': 'Loss',
                }
            }
        }
        return tmp
    def __MIF(self):
        tmp = {
            'commonParas':{
                'need_data_aligned': False,
                'need_model_aligned': False,
                'use_finetune': True,
                'use_bert': True,
                'early_stop': 8,
                'update_epochs': 2,
                'rnncell': 'gru',
                'use_cmd_sim': True,
            },
            # dataset
            'datasetParas':{
                'mosi':{
                    # the batch_size of each epoch is updata_epochs * batch_size
                    'batch_size': 64,
                    'learning_rate': 1e-4,
                    'hidden_size': 128,
                    'dropout': 0.2,
                    'attention_dropout':  0.2,
                    'reverse_grad_weight': 0.8,
                    'diff_weight': 0.3,
                    'sim_weight': 0.8,
                    'sp_weight': 0.0,
                    # when grad_clip == -1.0, means not use that
                    'grad_clip': 1.0,
                    'weight_decay': 0.002,
                    'num_head':  8,
                },
                'mosei':{
                    # the batch_size of each epoch is updata_epochs * batch_size
                    'batch_size': 32,
                    'learning_rate': 1e-4,
                    'hidden_size': 128,
                    'dropout': 0.2,
                    'reverse_grad_weight': 0.5,
                    'diff_weight': 0.1,
                    'sim_weight': 1.0,
                    'sp_weight': 1.0,
                    # when grad_clip == -1.0, means not use that
                    'grad_clip': 0.8,
                    'weight_decay': 0.0,
                },
                'sims':{
                    # the batch_size of each epoch is updata_epochs * batch_size
                    'batch_size': 64,
                    'learning_rate': 1e-4,
                    'hidden_size': 128,
                    'dropout': 0.5,
                    'attention_dropout':  0.2,
                    'reverse_grad_weight': 0.5,
                    'diff_weight': 0.3,
                    'sim_weight': 0.8,
                    'sp_weight': 1.0,
                    # when grad_clip == -1.0, means not use that
                    'grad_clip': -1.0,
                    'weight_decay': 0,
                    'num_head':  8,
                    'Decoder_head': 2,

                }
            },
        }
        return tmp
    
    def get_config(self):
        return self.args

if __name__ == "__main__":
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--debug_mode', type=bool, default=False,
                            help='adjust parameters ?')
        parser.add_argument('--modelName', type=str, default='mif',
                            help='support mif')
        parser.add_argument('--datasetName', type=str, default='sims',
                            help='support mosi/sims')
        parser.add_argument('--tasks', type=str, default='M',
                            help='M/T/A/V/MTAV/...')
        parser.add_argument('--num_workers', type=int, default=12,
                            help='num workers of loading data')
        parser.add_argument('--model_save_path', type=str, default='results/model_saves',
                            help='path to save model.')
        parser.add_argument('--res_save_path', type=str, default='results/result_saves',
                            help='path to save results.')
        parser.add_argument('--data_dir', type=str, default='./data/dataset/multimodal-sentiment-dataset',
                            help='path to data directory')
        parser.add_argument('--gpu_ids', type=list, default=[0,1],
                            help='indicates the gpus will be used.')
        return parser.parse_args()
        
    args = parse_args()
    config = ConfigTune(args)
    args = config.get_config()
    print(args)