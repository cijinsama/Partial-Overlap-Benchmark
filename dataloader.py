import os
import datasets
import pandas as pd
import os
from textgrid import TextGrid
from g2p_en import G2p
import random

GOOGLE_CORPUS_PATH = '/home/yi/data/google-10000-english.txt'
MAX_PHONEME_LEN = 25

def hasattrandtrue(o, name):
    return hasattr(o, name) and getattr(o, name)

def parse_textgrid_phonemes(file_path, tier_name='phones', target='phoneme'):
    tg = TextGrid.fromFile(file_path)
    phoneme_tier = next((t for t in tg.tiers if t.name.lower() == tier_name.lower()), None)

    if phoneme_tier is None:
        raise ValueError(f"No tier named '{tier_name}' found in TextGrid.")

    phoneme_list = []
    for interval in phoneme_tier.intervals:
        label = interval.mark.strip()
        if label:
            phoneme_list.append({
                'start': interval.minTime,
                'end': interval.maxTime,
                target: label
            })

    return phoneme_list

def load_POBLP(root, tokenizer):
    folders = [os.path.join(root, name) for name in os.listdir(root) if os.path.isdir(os.path.join(root, name))]
    datasetdict = {}
    g2p = G2p()
    with open(GOOGLE_CORPUS_PATH, encoding="utf-8") as f:
        top10k_words = [line.strip().lower() for line in f if line.strip()]

    def safe_float_list_to_int(lst):
        try:
            return [int(x) for x in lst]
        except (ValueError, TypeError):
            return None

    for folder in folders:
        csvs = [os.path.join(folder, name) for name in os.listdir(folder) if name.endswith('.csv')]
        all_df = []
        for csv in csvs:
            df = pd.read_csv(csv)
            ndf = pd.DataFrame()
            ndf['query_audio'] = df['comparison'].apply(lambda x: os.path.join(os.path.dirname(csv), x))
            ndf['query_audio_textgrid'] = ndf['query_audio'].apply(lambda x: x.replace('.wav', '.TextGrid'))
            ndf['query_text'] = df['comparison_text'].map(g2p).map(lambda x: [item for item in x if item.strip() != '']).map(tokenizer.encode)
            ndf['query_transcript'] = df['comparison_text']
            ndf['query_text_len'] = ndf['query_text'].map(len)
            ndf['anchor_audio'] = df['anchor'].apply(lambda x: os.path.join(os.path.dirname(csv), x))
            ndf['anchor_audio_textgrid'] = ndf['anchor_audio'].apply(lambda x: x.replace('.wav', '.TextGrid'))
            ndf['match_label'] = df['target']
            ndf['type'] = df['type']

            def process_anchor_text(row):
                text = str(row['anchor_text']).strip()
                words = text.split()
                if row['target'] == 1:
                    insert_word = random.choice(top10k_words)
                    while insert_word in words:
                        insert_word = random.choice(top10k_words)
                    new_words = words + [insert_word]
                    return " ".join(new_words), 0
                return None, 0

            ndf[['anchor_transcript', 'match_label']] = df.apply(process_anchor_text, axis=1, result_type='expand')
            ndf = ndf.dropna(subset=['anchor_transcript'])

            ndf['anchor_text'] = ndf['anchor_transcript'].map(g2p).map(lambda x: [item for item in x if item.strip() != '']).map(tokenizer.encode)
            ndf['anchor_text_len'] = ndf['anchor_text'].map(len)
            ndf = ndf[ndf['anchor_text_len'] <= MAX_PHONEME_LEN]

            ndf['query_text'] = ndf['query_text'].apply(safe_float_list_to_int)
            ndf['anchor_text'] = ndf['anchor_text'].apply(safe_float_list_to_int)
            ndf = ndf.dropna(subset=['anchor_text', 'query_text'])

            all_df.append(ndf)

        dataset = datasets.Dataset.from_pandas(pd.concat(all_df, ignore_index=True))
        dataset = dataset.cast_column("query_audio", datasets.Audio())
        dataset = dataset.cast_column("anchor_audio", datasets.Audio())
        dataset = dataset.cast_column("match_label", datasets.Value("int64"))
        datasetdict[os.path.basename(folder)] = dataset
    
    return datasets.DatasetDict(datasetdict)

def load_POBSP(root, tokenizer):
    folders = [os.path.join(root, name) for name in os.listdir(root) if os.path.isdir(os.path.join(root, name))]
    g2p = G2p()
    def safe_float_list_to_int(lst):
        try:
            return [int(x) for x in lst]
        except (ValueError, TypeError):
            return None
    datasetdict = {}
    for folder in folders:
        csv = os.path.join(folder, 'meta.csv')
        df = pd.read_csv(csv)
        df = df[~df['query_audio'].isna()]
        df = df[~df['anchor_audio'].isna()]
        ndf = pd.DataFrame()
        ndf['query_audio'] = df['query_audio'].apply(lambda x: os.path.join(os.path.dirname(csv), x))
        ndf['query_audio_textgrid'] = ndf['query_audio'].apply(lambda x: x.replace('.wav', '.TextGrid'))
        ndf['query_text'] = df['query_text'].map(g2p).map(lambda x: [item for item in x if item.strip() != '']).map(tokenizer.encode)
        ndf['query_transcript'] = df['query_text']
        ndf['query_text_len'] = ndf['query_text'].map(len)
        ndf['anchor_audio'] = df['anchor_audio'].apply(lambda x: os.path.join(os.path.dirname(csv), x))
        ndf['anchor_audio_textgrid'] = ndf['anchor_audio'].apply(lambda x: x.replace('.wav', '.TextGrid'))
        ndf['anchor_text'] = df['anchor_text'].map(g2p).map(lambda x: [item for item in x if item.strip() != '']).map(tokenizer.encode)
        ndf['anchor_transcript'] = df['anchor_text']
        ndf['anchor_text_len'] = ndf['anchor_text'].map(len)
        ndf['match_label'] = (ndf['anchor_text'] == ndf['query_text']).astype(int)

        ndf['query_text'] = ndf['query_text'].apply(safe_float_list_to_int)
        ndf['anchor_text'] = ndf['anchor_text'].apply(safe_float_list_to_int)
        ndf = ndf.dropna(subset=['anchor_text', 'query_text'])


        dataset = datasets.Dataset.from_pandas(ndf)
        dataset = dataset.cast_column("query_audio", datasets.Audio())
        dataset = dataset.cast_column("anchor_audio", datasets.Audio())
    
        datasetdict[os.path.basename(folder)] = dataset
    
    return datasets.DatasetDict(datasetdict)
