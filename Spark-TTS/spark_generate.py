import sys
import os
import torch
import numpy as np
import soundfile as sf
import logging
from datetime import datetime
import time
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cli.SparkTTS import SparkTTS

def generate_tts_audio(
    text,
    model_dir,
    device="cuda:1",
    prompt_speech_path=None,
    prompt_text=None,
    gender=None,
    pitch=None,
    speed=None,
    emotion=None,
    save_dir="example/results",
    segmentation_threshold=150,
    seed=None,
    model=None,
    skip_model_init=False,
):
    """
    Generates TTS audio from input text, splitting into segments if necessary.

    Args:
        text (str): Input text for speech synthesis.
        model_dir (str): Path to the model directory.
        device (str): Device identifier (e.g., "cuda:0" or "cpu").
        prompt_speech_path (str, optional): Path to prompt audio for cloning.
        prompt_text (str, optional): Transcript of prompt audio.
        gender (str, optional): Gender parameter ("male"/"female").
        pitch (str, optional): Pitch parameter (e.g., "moderate").
        speed (str, optional): Speed parameter (e.g., "moderate").
        emotion (str, optional): Emotion tag (e.g., "HAPPY", "SAD", "ANGRY").
        save_dir (str): Directory where generated audio will be saved.
        segmentation_threshold (int): Maximum number of words per segment.
        seed (int, optional): Seed value for deterministic voice generation.

    Returns:
        str: The unique file path where the generated audio is saved.
    """
    # ============================== OPTIONS REFERENCE ==============================
    # ✔ Gender options: "male", "female"
    # ✔ Pitch options: "very_low", "low", "moderate", "high", "very_high"
    # ✔ Speed options: same as pitch
    # ✔ Emotion options: list from token_parser.py EMO_MAP keys
    # ✔ Seed: any integer (e.g., 1337, 42, 123456) = same voice (mostly)
    # ==============================================================================

    global _model_cache

    device_key = str(device)
    if not skip_model_init or model is None:
        if device_key not in _model_cache:
            logging.info(f"Initializing TTS model on {device_key}...")
            if not prompt_speech_path:
                logging.info(
                    f"Using Gender: {gender or 'default'}, Pitch: {pitch or 'default'}, Speed: {speed or 'default'}, Emotion: {emotion or 'none'}, Seed: {seed or 'random'}"
                )
            model = SparkTTS(model_dir, torch.device(device))
            _model_cache[device_key] = model
        else:
            model = _model_cache[device_key]

    # Set seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logging.info(f"Seed set to: {seed}")

    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]
    save_path = os.path.join(
        save_dir,
        f"{'-'.join(text.split(' '))}_{gender}-{pitch}-{speed}-{timestamp}.wav",
    )

    words = text.split()
    if len(words) > segmentation_threshold:
        logging.info("Text exceeds threshold; splitting into segments...")
        segments = [
            " ".join(words[i : i + segmentation_threshold])
            for i in range(0, len(words), segmentation_threshold)
        ]
        wavs = []
        for seg in segments:
            with torch.no_grad():
                wav = model.inference(
                    seg,
                    prompt_speech_path,
                    prompt_text=prompt_text,
                    gender=gender,
                    pitch=pitch,
                    speed=speed,
                )
            wavs.append(wav)
        final_wav = np.concatenate(wavs, axis=0)
    else:
        with torch.no_grad():
            final_wav = model.inference(
                text,
                prompt_speech_path,
                prompt_text=prompt_text,
                gender=gender,
                pitch=pitch,
                speed=speed,
            )

    sf.write(save_path, final_wav, samplerate=16000)
    # logging.info(f"Audio saved at: {save_path}")
    return save_path

import random


import pandas as pd
import sys
def main(csv_path, gpu_id, model_dir, save_dir):
    genders = ["male", "female"]
    pitch = "moderate"
    speed = "moderate"

    df = pd.read_csv(csv_path)
    COMMANDS = {}
    for x in df['query_text']:
        COMMANDS[x] = None
    for x in df['anchor_text']:
        COMMANDS[x] = None

    device_str = f"cuda:{gpu_id}"
    model = None

    logging.info(f"Initializing TTS model on {device_str}...")
    model = SparkTTS(model_dir, torch.device(device_str))

    for text in tqdm(COMMANDS):
        try:
            output_file = generate_tts_audio(
                text=text,
                model_dir=model_dir,
                gender=random.choice(genders),
                pitch=pitch,
                speed=speed,
                emotion=None,
                seed=random.randint(0, 1000000),
                prompt_speech_path=None,
                prompt_text=None,
                save_dir=save_dir,
                device=device_str,
                model=model,
                skip_model_init=True,
            )
            print(f"Generated audio for '{text}' on GPU {gpu_id}: {output_file}")
            COMMANDS[text] = output_file
        except Exception as e:
            print(f"Error on GPU {gpu_id} for text '{text}': {e}")
            COMMANDS[text] = None

    query_audio_paths = []
    anchor_audio_paths = []

    for idx, row in df.iterrows():
        q_text = row['query_text']
        a_text = row['anchor_text']

        query_audio_paths.append(COMMANDS.get(q_text, None))
        anchor_audio_paths.append(COMMANDS.get(a_text, None))

    df['query_audio'] = query_audio_paths
    df['anchor_audio'] = anchor_audio_paths

    output_path = os.path.join(save_dir, os.path.basename(csv_path) + ".meta.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved output csv to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python your_script.py <csv_path> <gpu_id> <model_dir> <save_dir>")
        sys.exit(1)

    csv_path = sys.argv[1]
    gpu_id = int(sys.argv[2])
    model_dir = sys.argv[3]
    save_dir = sys.argv[4]

    main(csv_path, gpu_id, model_dir, save_dir)
