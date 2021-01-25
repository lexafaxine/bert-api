python bert_ner.py --task_name="NER" \
--do_train=True \
--do_eval=True \
--do_predict=True \
--data_dir=./HironsanData \
--vocab_file=./checkpoint/vocab.txt \
--bert_config_file=./checkpoint/config.json \
--init_checkpoint=./checkpoint/model.ckpt \
--max_seq_length=128 \
--output_dir=./output/result_dir/ \
--num_train_epochs=100