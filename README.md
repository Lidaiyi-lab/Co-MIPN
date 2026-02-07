# Title
Co-MIPN: Cross-modal guided interactive entity recognition network via contrastive aggregation and dynamic similarity 

Description - An overview of the code/dataset
==========
(1) An overview of the dataset
==========
1. The repository is validated on two standard MNER benchmark datasets: Twitter2015 and Twitter2017, which are widely used in multimodal information extraction research. Both datasets are derived from Twitter posts, combining short text and associated images to address entity recognition under real-world social media scenarios.
2. The text data follows the conll format. You can download the Twitter2015 data the Twitter2017 data via this link (https://doi.org/10.6084/m9.figshare.31123498). Please place them in data/NER_data.
3. You can also put them anywhere and modify the path configuration in run.py ‎Requirements – Any dependencies (e.g., Python libraries).

(2) An overview of the code
==========
The expected structure of files is:

```shell
 |-- data
 |    |-- NER_data
 |    |    |-- twitter2015  # text data
 |    |    |    |-- train.txt
 |    |    |    |-- valid.txt
 |    |    |    |-- test.txt
 |    |    |    |-- twitter2015_train_dict.pth
 |    |    |    |-- ...
 |    |    |-- twitter2015_images       # raw image data
 |    |    |-- twitter2015_aux_images   # object image data
 |    |    |-- twitter2017
 |    |    |-- twitter2017_images
 |    |    |-- twitter2017_aux_images
 |-- models	# models
 |    |-- bert_model.py
 |    |-- modeling_bert1.py
 |    |-- modeling_bert2.py
 |    |-- modeling_bert3.py
 |    |-- modeling_bert4.py
 |-- modules
 |    |-- metrics.py    # metric
 |    |-- train.py  # trainer
 |-- processor
 |    |-- dataset.py    # processor, dataset
 |-- logs     # code logs
 |-- run.py   # main 
 |-- run_ner_task.sh
```
Code Information
==========
1. models/bert_model.py: Integrates all key components (contrastive aggregation, dynamic matching, cross-modal fusion) into a unified framework.
2. models/metrics.py: Defines evaluation metrics for NER tasks (e.g., precision, recall, F1-score).
3. processor/dataset.py: Loads text (tokenization via BERT tokenizer) and visual data (feature extraction via CLIP/DINO), and aligns them using pre-built mapping dictionaries.
4. train/trainer.py: Implements the training process with combined loss (contrastive loss + NER CRF loss) and supports Adam optimizer, learning rate scheduling, and model checkpoint saving.
5. run.py: Supports flexible parameter configuration (e.g., --use_contrastive for enabling contrastive aggregation, --use_dynamic_matching for dynamic similarity matching) to adapt to different experimental settings.
   
Data Information
==========
+ Twitter2015 & Twitter2017

Data download: https://doi.org/10.6084/m9.figshare.31123498, Please place them in `data/NER_data`.

Requirements
==========
To run the codes, you need to install the requirements:
```
pip install -r requirements.txt
```
‎Usage Instructions 
==========
Data Processing
==========
To extract visual object images, we first use the NLTK parser to extract noun phrases from the text and apply the [visual grouding toolkit](https://github.com/zyang-ur/onestage_grounding) to detect objects. Detailed steps are as follows:

1. Using the NLTK parser (or Spacy, textblob) to extract noun phrases from the text.
2. Applying the [visual grouding toolkit](https://github.com/zyang-ur/onestage_grounding) to detect objects. Taking the twitter2015 dataset as an example, the extracted objects are stored in `twitter2015_aux_images`. The images of the object obey the following naming format: `imgname_pred_yolo_crop_num.png`, where `imgname` is the name of the raw image corresponding to the object, `num` is the number of the object predicted by the toolkit. (Note that in `train/val/test.txt`, text and raw image have a one-to-one relationship, so the `imgname` can be used as a unique identifier for the raw images)
3. Establishing the correspondence between the raw images and the objects. We construct a dictionary to record the correspondence between the raw images and the objects. Taking `twitter2015/twitter2015_train_dict.pth` as an example, the format of the dictionary can be seen as follows: `{imgname:['imgname_pred_yolo_crop_num0.png', 'imgname_pred_yolo_crop_num1.png', ...] }`, where key is the name of raw images, value is a List of the objects.

The detected objects and the dictionary of the correspondence between the raw images and the objects are available in our data links.

Data Download
==========

+ Twitter2015 & Twitter2017

    The text data follows the conll format. You can download the Twitter2015 data the Twitter2017 data via this link (https://doi.org/10.6084/m9.figshare.31123498). Please place them in data/NER_data.
    You can also put them anywhere and modify the path configuration in `run.py`

Train
==========

## NER Task

The data path and GPU related configuration are in the `run.py`. To train ner model, run this script.

```shell
bash run_twitter15.sh
bash run_twitter17.sh
```

Test
==========
## NER Task

To test ner model, you can use the tained model and set `load_path` to the model path, then run following script:

```shell
python -u run.py \
      --dataset_name="twitter15/twitter17" \
      --bert_name="bert-base-uncased" \
      --seed=1234 \
      --only_test \
      --max_seq=192 \
      --use_prompt \
      --prompt_len=12 \
      --sample_ratio=1.0 \
      --load_path='your_ner_ckpt_path'

```
Acknowledgement
==========

The acquisition of Twitter15 and Twitter17 data refer to the code from [UMT](https://github.com/jefferyYu/UMT/), many thanks.

Citations
==========
Jianfei Yu, Jing Jiang, Li Yang, and Rui Xia. 2020. Improving Multimodal Named Entity Recognition via Entity Span Detection with Unified Multimodal Transformer. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 3342–3352.

