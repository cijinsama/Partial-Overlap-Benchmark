# Recipe for Partial-Overlap-Benchmark
[No Word Left Behind: Mitigating Prefix Bias in Open-Vocabulary Keyword Spotting (ICASSP 2026)](http://google.com)

:loudspeaker: [09/16/25] Our paper was submitted to ICASSP-2026
## About the dataset  
The **Partial Overlap Benchmark (POB)** is introduced to evaluate open-vocabulary keyword spotting (OV-KWS) models in scenarios where queries and enrolled phrases share overlapping prefixes, but not the same (e.g., *“turn the light on”* vs. *“turn the light off”*). Existing benchmarks like LibriPhrase and Google Speech Commands rarely test such cases, leading to **prefix bias** in current models.  

POB consists of two complementary datasets:  

* **POB-LibriPhrase (POB-LP)** – derived from LibriPhrase by appending common English words to enforce prefix overlaps.  
* **POB-Spark** – a synthetic corpus generated using the [Spark-TTS model](https://github.com/SparkAudio/Spark-TTS) to provide controlled overlap patterns across diverse speaker characteristics.  

### What are POB cases
| Enrolled phrase | Overlapping query | Result |  
|----|----|----|  
| turn the light on | turn the light off | **prefix confusable** |  
| service | survive | **not prefix confusable** |  
| play the music | stop the music | **not prefix confusable** |  

## Getting started
You can either load data directly from our processed huggingface repo following the [instruction](#Load-from-huggingface) or generate the data by yourself following the [instruction](#Generate-from-scratch).

### Load from huggingface
```
from datasets import load_dataset

dataset = load_dataset("RiceHunger/Partial-Overlap-Benchmark")
print(dataset["LP"][0])
```
You can get
```python
{'query_audio': {'path': '5405-121045-0041_0.wav',
  'array': array([ 0.00109863,  0.00146484,  0.00137329, ..., -0.0015564 ,
         -0.00149536, -0.00247192], shape=(10240,)),
  'sampling_rate': 16000},
 'query_audio_textgrid': 'YOUR_PATH/LibriPhrase/evaluation_set/train-other-500/train-other-500/5405/121045/5405-121045-0041_0.TextGrid',
 'query_text': [42, 5, 54, 35, 41],
 'query_transcript': 'carriage',
 'query_text_len': 5,
 'anchor_audio': {'path': '1006-135212-0068_0.wav',
  'array': array([0.08691406, 0.08447266, 0.08007812, ..., 0.03323364, 0.03277588,
         0.0329895 ], shape=(8320,)),
  'sampling_rate': 16000},
 'anchor_audio_textgrid': 'YOUR_PATH/LibriPhrase/evaluation_set/train-other-500/train-other-500/1006/135212/1006-135212-0068_0.TextGrid',
 'anchor_text': [42, 5, 54, 35, 41, 42, 14, 45, 57, 35, 46],
 'anchor_transcript': 'carriage counting',
 'anchor_text_len': 11,
 'match_label': 0}
```


### Generate from scratch
#### 0. Environment
```
conda create --name pob python==3.12.0
pip install -r requirements.txt
```
#### 1. Preparation
Prepare [LibriPhrase](https://github.com/gusrud1103/LibriPhrase) by following their instructions.
After processing, your directory should look like this:
```
YOUR_SAVE_DIR
├── train_500
│   ├── libriphrase_diffspk_all_train500_1word.csv
│   ├── libriphrase_diffspk_all_train500_2word.csv
│   ├── libriphrase_diffspk_all_train500_3word.csv
│   ├── libriphrase_diffspk_all_train500_4word.csv
│   └── train-other-500
├── train_360
│   ├── libriphrase_diffspk_all_train360_1word.csv
│   ├── libriphrase_diffspk_all_train360_2word.csv
│   ├── libriphrase_diffspk_all_train360_3word.csv
│   ├── libriphrase_diffspk_all_train360_4word.csv
│   └── train-clean-360
└── train_100
    ├── libriphrase_diffspk_all_train100_1word.csv
    ├── libriphrase_diffspk_all_train100_2word.csv
    ├── libriphrase_diffspk_all_train100_3word.csv
    ├── libriphrase_diffspk_all_train100_4word.csv
    └── train-clean-100

```
#### 2. Generating POB-Spark datasets
POB-Spark construction steps:
1. **Word → phoneme mapping** using the CMU dictionary.  
2. **Nearest phonetic neighbor search** using Levenshtein distance in phoneme space.  
3. **Phrase construction** by sampling words under a phoneme-length constraint.  
4. **Word replacement** to create confusable pairs.  
5. **Sample selection** to ensure a roughly uniform distribution of the first-differing index across phrase lengths.  
6. **Finalize dataset** by adding (q, a, False), (a, q, False), and (q, q, True) pairs.
7. **Use Spark-TTS** to generate corresponding audio

Run the script to generate meta data:
```
python prepare_meta.py --num_perposition 100 --num_pairs 50000 --max_len 25 --output meta_text.csv
```
Set up Spark-TTS according to [link](https://github.com/SparkAudio/Spark-TTS) and synthesize audio:
```
cd Spark-TTS; python spark_generate.py ../meta_text.csv YOUR_PRETRAINED_MODEL_DIR/Spark-TTS-0.5B YOUR_SAVE_DIR/valid
```
Run MFA for forced alignment:
```
find YOUR_SAVE_DIR/valid -type f -name "*.wav" | while read -r file; do filename=$$(basename $$file .wav); first_part=$${filename%%_*}; processed=$$(echo $$first_part | tr '-' ' ' | tr 'A-Z' 'a-z'); echo "$$processed" > "$${file%.wav}.txt"; done
mfa align YOUR_SAVE_DIR/valid english_us_arpa english_us_arpa YOUR_SAVE_DIR/valid --single_speaker
```
After alignment, your folder should look like:
```
YOUR_SAVE_DIR
└── valid
    ├── meta.csv
    ├── *.txt
    ├── *.wav
    ├── *.TextGrid
    ...
```
#### 3. Load the data(POB-LP will be generated on the fly)
Update file paths in `configs/data/*.yaml` and `dataloader.py`
```yaml
# *.yaml
valid: YOUR_DIR/POBLP # change this to your path
cache_dir: YOUR_DIR/datasets_cache # change this to your path
```

```python
# dataloader.py
GOOGLE_CORPUS_PATH = '/home/yi/data/google-10000-english.txt' # change this to your path
```
Finally, load the data in `example.py`
```python
python example.py
```

---

## Reference
Yi Liu, Chuan-Che (Jeff) Huang, Xiao Quan.  
**“No Word Left Behind: Mitigating Prefix Bias in Open-Vocabulary Keyword Spotting.”**  
ICASSP 2026.

## License

## Citation
If you use this dataset or code, please cite:

    @inproceedings{liu2026pob,
        author={Yi Liu and Chuan-Che Huang and Xiao Quan},
        title={No Word Left Behind: Mitigating Prefix Bias in Open-Vocabulary Keyword Spotting},
        booktitle={ICASSP 2026},
        year={2026}
    }
