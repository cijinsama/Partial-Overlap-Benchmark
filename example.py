import dataloader
import transformers

LP_PATH = '/home/yi/data/LibriPhrase'
SP_PATH = '/home/yi/data/POBSP'
VOCAB_PATH = './vocab.json'

if __name__ == '__main__':
    tokenizer = transformers.Wav2Vec2PhonemeCTCTokenizer(VOCAB_PATH)
    valid_ds = dataloader.load_POBSP(LP_PATH, tokenizer)
    print("Dataset length:", len(valid_ds))
    valid_ds = dataloader.load_POBLP(LP_PATH, tokenizer)
    print("Dataset length:", len(valid_ds))
