task_name:
    目前只支持4种
    目前只有这四种
    cola
    mnli
    mrpc
    xnli
data_dir:
    训练测试文件的文件夹
    只支持csv文件

# bert分类问题
python bert_base/bert/run_classifier.py \
    --data_dir=/home/wanglei/algorithm/github/bert-ner-bilstm-crf/data_set \
    --task_name=cola \
    --vocab_file=/home/wanglei/algorithm/github/bert-ner-bilstm-crf/bert_base/chinese_L-12_H-768_A-12/vocab.txt \
    --bert_config_file=/home/wanglei/algorithm/github/bert-ner-bilstm-crf/bert_base/chinese_L-12_H-768_A-12/bert_config.json \
    --output_dir=/home/wanglei/algorithm/github/bert-ner-bilstm-crf/output \
    --do_train=True

# 训练ner的代码
bert-base-ner-train \
    -data_dir /home/wanglei/algorithm/github/bert-bilstm-crf-ner/NERdata \
    -output_dir /home/wanglei/algorithm/github/bert-bilstm-crf-ner/train_data \
    -init_checkpoint /home/wanglei/algorithm/github/bert-bilstm-crf-ner/bert_base/chinese_L-12_H-768_A-12/bert_model.ckpt \
    -bert_config_file /home/wanglei/algorithm/github/bert-bilstm-crf-ner/bert_base/chinese_L-12_H-768_A-12/bert_config.json \
    -vocab_file /home/wanglei/algorithm/github/bert-bilstm-crf-ner/bert_base/chinese_L-12_H-768_A-12/vocab.txt \
    -max_seq_length 512


