# Sign2Vis: Automated Data Visualization from Sign Language

## Dataset information
- Some samples in our dataset

**Natural Language:**  Bar chart showing the sum of meters by nationality in descending order of the y-axis.  

**Partial frame of Pose video:**  

![image](samples/2908@y_name@DESC.png)

**Query:** mark bar data swimmer encoding x nationality y aggregate sum meter_100 transform group x sort y desc  

**Natural Language:** Line chart showing the number of creations over time and binning by year

**Partial frame of Pose video:**  

![image](samples/1093.png)

**Query:** mark line data department encoding x creation y aggregate count creation transform bin x by year  

We release the text2vis dataset in  `./text_refinement/s2v_data.jsonl.` The video dataset is so huge(200~300GB) that we haven't find a proper way to release it yet.

## Requirements

- `python3.6` or higher.
- `PyTorch 1.7.1` or higher.
- `torchtext 0.8`
- `ipyvega`
- The pre-trained BERT parameters are available at [here](https://drive.google.com/file/d/1iJvsf38f16el58H4NPINQ7uzal5-V4v4/view?usp=sharing)

## Data Process
**Note： for executing the code, you may need to modify the input and output path in our code.**
- convert the sign videos(mp4 format) in npy format
    ```shell
    cd sign2vis/dataset
    python sign_preprocess.py
    ```

- split and annotate dataset
    ```shell
    cd ./text_refinement
    python split_dataset.py
    ```

## Train
- train the Sign2Text model（Transformer）
    ```shell
    CUDA_VISIBLE_DEVICES=0 nohup python -u train_sign2text.py --do_train --seed 1 --bS 4 --accumulate_gradients 2 --bert_type_abb uS --lr 0.0001 > sign2text.log 2>&1 &
    ```

- train the Text2VIS model（ncNet）
    ```shell
    cd ./ncNet
    CUDA_VISIBLE_DEVICES=0 nohup python -u train.py  > train.log 2>&1 &
    ```

- after finish the above model training，inference with above two models（Transformer+ncNet）
    ```shell
    CUDA_VISIBLE_DEVICES=0 nohup python -u test_sign2text.py --seed 1 --bS 4 --bert_type_abb uS --trained > sign2text.log 2>&1 &
    cd ./ncNet
    CUDA_VISIBLE_DEVICES=0 nohup python -u test.py > test.log
    ```

- train the end-to-end model（Sign2VisNet）
    ```shell
    CUDA_VISIBLE_DEVICES=0 nohup python -u train_sign2vis.py --with_temp 2 --batch_size 4 --accumulate_gradients 2 --bert_type_abb uS >> ./train_sign2vis.log 2>&1 &
    ```

