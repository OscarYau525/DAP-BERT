# DAP-BERT

# Installation
This repo is tested on pytorch 1.6.0.
Download the GLUE dataset using [download_glue_data.py](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e), i.e. run
```
python download_glue_data.py --data_dir glue_data --tasks all
```

# Data augmentation
1. Get pre-trained bert model from [huggingface](https://huggingface.co/bert-base-uncased/tree/main).
2. Get GloVe word embeddings [glove.42B.300d.txt](https://nlp.stanford.edu/data/glove.42B.300d.zip).
3. Run 
    ```
    python data_augmentation.py --pretrained_bert_model ${BERT_BASE_DIR} \
        --glove_embs ${GLOVE_EMB} \
        --glue_dir ${GLUE_DIR} \  
        --task_name ${TASK_NAME}
    ```

# Prepare your fine-tuned model
Depending on the resource constraint, choose
1. fine-tuning a bert-base-uncased by scripts/fine_tune.sh, e.g.
    ```
    bash scripts/ft_bertbase.sh CoLA bert-base-uncased 10 42
    ```

2. or download fine-tuned models from [TinyBERT](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT), i.e. TinyBERT(4layer-312dim) or TinyBERT(6layer-768dim)

Please put all models in DAP-BERT/models.

# Demo
Edit scripts/demo.sh to change paths to fine-tuned models and GLUE dataset. 
Example to run demo.sh:
```
bash scripts/demo.sh 0.5 0.1 CoLA 10 10 42
```

