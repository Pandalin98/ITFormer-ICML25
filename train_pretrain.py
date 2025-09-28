import warnings
warnings.simplefilter("ignore", category=UserWarning)
from transformers import  AutoTokenizer
from transformers import AutoProcessor
import torch.nn as nn
from transformers import Trainer
from dataset.dataset import TsQaDataset,PretrainDataset
import argparse
from models.TimeLanguageModel import TLMConfig
import swanlab as wandb
from EXP.exp_pretraining import Exp_Pretrain


if __name__ == '__main__':

    #读取args
    parser = argparse.ArgumentParser(description='TsEncoder Pretrain')
    parser.add_argument('--fix_seed', type=int, default=None, help='seed')

    #Model  settings
    parser.add_argument('--model', type=str, required=False, default='TimeSeriesEncoder',
                        help='model name')
    parser.add_argument('--d_model', type=int, default=512,
                        help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=4,
                        help='num of encoder layers')
    parser.add_argument("--patch_len", type=int, default=60)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--input_len", type=int, default=600)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')

    #Pretrain settings
    parser.add_argument('--pretrain', type=bool, default=True, help='pretrain mode')
    parser.add_argument('--min_mask_ratio', type=float, default=0.7, help='minimum mask ratio')
    parser.add_argument('--max_mask_ratio', type=float, default=0.8, help='maximum mask ratio')

    # Training arguments
    parser.add_argument('--do_train', type=bool, default=True, help='whether to do training')
    parser.add_argument('--per_device_train_batch_size', type=int, default=12, help='batch size per device during training')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=12, help='batch size for evaluation')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='number of updates steps to accumulate before performing a backward/update pass')
    parser.add_argument('--num_train_epochs', type=int, default=10, help='number of training epochs')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')

    #Efficiency settings
    parser.add_argument('--fp16', type=bool, default=True, help='whether to use 16-bit (mixed) precision')
    parser.add_argument('--dataloader_pin_memory', type=bool, default=True, help='pin memory in data loader')
    parser.add_argument('--dataloader_num_workers', type=int, default=8, help='number of subprocesses to use for data loading')

    #logging settings
    parser.add_argument('--output_dir', type=str, default='save/pretrain_ts_small', help='output directory')
    parser.add_argument('--save_steps', type=int, default=100, help='save checkpoint every X updates steps')
    parser.add_argument('--save_total_limit', type=int, default=2, help='limit the total amount of checkpoints')
    parser.add_argument('--logging_steps', type=int, default=10, help='log every X updates steps')
    parser.add_argument('--report_to', type=str, default="swanlab", help='report results to')

    args = parser.parse_args()

    ##data setting
    tlmconfig = TLMConfig(llm_model_path = 'LLM/Qwen2.5-0.5B-Instruct')
    ts_path = 'dataset/dataset_for_upload_modelscope/data_merged_new.h5'
    tokenizer = AutoTokenizer.from_pretrained(tlmconfig.llm_model_path)
    processor = AutoProcessor.from_pretrained(tlmconfig.llm_model_path)
    dataset = PretrainDataset(ts_path)

    wandb.init(mode="offline",project="TSLLM-TsEncoder", name="XXX")

    Trainer = Exp_Pretrain(args, dataset)

    Trainer.train(resume_from_checkpoint=False)
    Trainer.save_model('save/pretrain')
    Trainer.save_state()