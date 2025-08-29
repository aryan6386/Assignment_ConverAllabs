# ------------------------------
# Call Quality Analyzer Script
# ------------------------------
# Features:
# 1. Upload OR Download audio (YouTube or local file)
# 2. Transcribe conversation
# 3. Perform speaker diarization (basic or pyannote if token is set)
# 4. Analyze talk-time, questions, monologues
# 5. Perform sentiment analysis
# 6. Provide actionable insight
# ------------------------------

# --- SECTION 1: IMPORTS ---
import os
import torch
import whisper
from transformers import pipeline
from google.colab import files

# Optional: for pyannote speaker diarization
try:
    from pyannote.audio import Pipeline
except ImportError:
    Pipeline = None

# --- SECTION 2: CHOOSE AUDIO SOURCE ---

USE_YOUTUBE = False  # 🔹 Set to True if downloading from YouTube, False to upload a local file
YOUTUBE_URL = ""
AUDIO_FILE_PATH = "sample.mp3"

if USE_YOUTUBE:
    print("\nDownloading audio from YouTube...")
    !yt-dlp --extract-audio --audio-format mp3 -o {AUDIO_FILE_PATH} {YOUTUBE_URL}
    print("✅ Audio download complete.")
else:
    print("\nUpload your local audio file:")
    uploaded = files.upload()
    filename = list(uploaded.keys())[0]
    AUDIO_FILE_PATH = filename
    print(f"✅ Uploaded: {AUDIO_FILE_PATH}")

# --- SECTION 3: TRANSCRIPTION ---
print("\nTranscribing audio with Whisper...")
model = whisper.load_model("base.en")
result = model.transcribe(AUDIO_FILE_PATH, fp16=False, language="en")
transcription_text = result['text']
segments = result['segments']
print("✅ Transcription complete.")

# --- SECTION 4: SPEAKER DIARIZATION ---
print("\nPerforming speaker diarization...")
HUGGINGFACE_TOKEN = None  # 🔹 Add token if you want real diarization
diarized_segments = []

if HUGGINGFACE_TOKEN and Pipeline is not None:
    pipeline_diarization = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=HUGGINGFACE_TOKEN)
    diarization_result = pipeline_diarization(AUDIO_FILE_PATH)

    for turn, _, speaker in diarization_result.itertracks(yield_label=True):
        diarized_segments.append({
            'speaker': speaker,
            'start': turn.start,
            'end': turn.end,
            'text': ""  # Mapping Whisper text would require alignment
        })
else:
    # Simplified diarization (alternate speakers)
    current_speaker = "Speaker A"
    for segment in segments:
        diarized_segments.append({
            'speaker': current_speaker,
            'start': segment['start'],
            'end': segment['end'],
            'text': segment['text']
        })
        current_speaker = "Speaker B" if current_speaker == "Speaker A" else "Speaker A"
print("✅ Diarization complete.")

# --- SECTION 5: CALL ANALYSIS ---
print("\nAnalyzing call data...")

speaker_talk_times = {}
questions_asked = 0
monologue_durations = {}
last_speaker = None

for segment in diarized_segments:
    talk_duration = segment['end'] - segment['start']
    speaker = segment['speaker']
    speaker_talk_times[speaker] = speaker_talk_times.get(speaker, 0) + talk_duration

    if segment['text'].strip().endswith('?'):
        questions_asked += 1

    if speaker not in monologue_durations:
        monologue_durations[speaker] = []

    if last_speaker == speaker:
        monologue_durations[speaker][-1] += talk_duration
    else:
        monologue_durations[speaker].append(talk_duration)

    last_speaker = speaker

all_monologues = [duration for spk in monologue_durations for duration in monologue_durations[spk]]
longest_monologue = max(all_monologues) if all_monologues else 0

# Sentiment analysis
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    tokenizer="distilbert-base-uncased"
)
sentiment_result = sentiment_analyzer(transcription_text)[0]
sentiment = sentiment_result['label'].lower()
sentiment_score = sentiment_result['score']

# --- SECTION 6: OUTPUT RESULTS ---
print("\n--- Call Analysis Report ---")

total_talk_time = sum(speaker_talk_times.values())
print("\n1. Talk-time Ratio:")
for speaker, duration in speaker_talk_times.items():
    ratio = (duration / total_talk_time) * 100 if total_talk_time > 0 else 0
    print(f"   - {speaker}: {ratio:.2f}%")

print(f"\n2. Number of Questions Asked: {questions_asked}")
print(f"\n3. Longest Monologue Duration: {longest_monologue:.2f} seconds")
print(f"\n4. Call Sentiment: {sentiment.capitalize()} (Score: {sentiment_score:.4f})")

print("\n5. One Actionable Insight:")
if total_talk_time > 0:
    speaker_list = list(speaker_talk_times.items())
    ratio_diff = abs(speaker_list[0][1] - speaker_list[1][1]) / total_talk_time if len(speaker_list) > 1 else 0

    if ratio_diff > 0.3:
        dominant_speaker = speaker_list[0][0] if speaker_list[0][1] > speaker_list[1][1] else speaker_list[1][0]
        print(f"   - Talk-time is unbalanced. '{dominant_speaker}' dominates the conversation.")
    elif sentiment == 'negative':
        print("   - Overall sentiment is negative. Investigate customer dissatisfaction.")
    elif questions_asked < 3:
        print("   - Few questions asked. Encourage more questions to understand customer needs.")
    else:
        print("   - Call is balanced and positive. Good work!")
else:
    print("   - Call too short to provide meaningful insight.")

print("\n--- Call Transcription ---")
for entry in diarized_segments:
    print(f"[{entry['start']:.2f}s] ({entry['speaker']}): {entry['text']}")

print("\nProcessing complete.")
