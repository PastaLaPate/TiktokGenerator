import os
import random

from moviepy.audio.AudioClip import CompositeAudioClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.VideoClip import TextClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.fx.speedx import *
from moviepy.video.fx.resize import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from scipy.io.wavfile import write as write_wav
from bark.generation import (
    generate_text_semantic,
    preload_models,
)
from bark.api import semantic_to_waveform
from bark import generate_audio, SAMPLE_RATE
import g4f
import nltk  # we'll use this to split into sentences


def selectMode():
    mode = input('Mode ? (render,script) ').lower()
    if mode == "render" or mode == "script" or mode == "auto":
        return mode
    else:
        return selectMode()


def generateSubtitles(content, audiosPath):
    sentences = nltk.sent_tokenize(content)
    i = 0
    start = 0
    clips = []
    for sentence in sentences:
        audioClip = AudioFileClip(audiosPath + f"/audio-{i}.wav")
        textClip = TextClip(txt=sentence, size=(600, 1800), method="caption", kerning=2, interline=1, stroke_color="black", fontsize=50, font="Arial-Bold", stroke_width=1, color="white")
        textClip = textClip.set_duration(audioClip.duration)
        textClip = textClip.set_start(start)
        start += audioClip.duration
        audioClip.close()
        textClip = textClip.set_position("center")
        clips.append(textClip)
        i += 1
    return clips

def randomGameplay():
    files = os.listdir('./gameplay')
    gameplay = files[random.randint(0, len(files)-1)]
    return f'./gameplay/{gameplay}'


def getBackground(content, audiosPath):
    duration = 0
    sentences = nltk.sent_tokenize(content)
    i = 0
    for sentence in sentences:
        audioClip = AudioFileClip(audiosPath + f"/audio-{i}.wav")
        duration += audioClip.duration
        audioClip.close()
        i += 1
    gameplay = randomGameplay()
    gameplay = VideoFileClip(gameplay)
    start = random.randint(0, int(gameplay.duration - duration))
    gameplay = resize(gameplay, newsize=(1080, 1920))
    gameplay = gameplay.subclip(start, start + duration)
    gameplay = gameplay.set_duration(duration)
    return gameplay

def getAudios(content, audiosPath):
    duration = 0
    sentences = nltk.sent_tokenize(content)
    i = 0
    clips = []
    for sentence in sentences:
        audioClip = AudioFileClip(audiosPath + f"/audio-{i}.wav")
        audioClip = audioClip.set_start(duration)
        duration += audioClip.duration
        clips.append(audioClip)
        i += 1
    return clips

def generateAudio(content, path, speaker="v2/en_speaker_0"):
    semantic_tokens = generate_text_semantic(content, history_prompt=speaker, temp=0.6, min_eos_p=0.05)
    audio_array = semantic_to_waveform(semantic_tokens, history_prompt=speaker)
    write_wav(path, SAMPLE_RATE, audio_array)

def getMusic(content, audiosPath):
    duration = 0
    sentences = nltk.sent_tokenize(content)
    i = 0
    for sentence in sentences:
        audioClip = AudioFileClip(audiosPath + f"/audio-{i}.wav")
        duration += audioClip.duration
        audioClip.close()
        i += 1
    gameplay = AudioFileClip("./musics/music.mp3")
    start = random.randint(0, int(gameplay.duration - duration))
    gameplay = gameplay.subclip(start, start + duration)
    gameplay = gameplay.set_duration(duration)
    return gameplay

def render(title):
    f = open(f'{title}/script.txt', "r")
    content = f.read()
    content = content.replace("\n", " ").replace("*", " ").strip()
    audiosPath = f'./{title}/audios'
    videoPath = f'./{title}/video/export.mp4'
    os.makedirs(audiosPath, exist_ok=True)
    sentences = nltk.sent_tokenize(content)
    speaker = "v2/en_speaker_0"
    part = 0
    for sentence in sentences:
        if not os.path.exists(f"{audiosPath}/audio-{part}.wav"):
            generateAudio(sentence, f"{audiosPath}/audio-{part}.wav")
        part += 1
    os.makedirs(f"./{title}/video", exist_ok=True)
    video = CompositeVideoClip([
        getBackground(content, audiosPath),
        *generateSubtitles(content, audiosPath)
    ]).set_audio(CompositeAudioClip(getAudios(content, audiosPath) + [getMusic(content, audiosPath)]))
    video = speedx(video, 1)
    video.write_videofile(videoPath, fps=30, preset="ultrafast")

def script(title, prompt):
    resp = g4f.ChatCompletion.create(
        model="gpt-3.5",
        provider=g4f.Provider.ChatBase,
        messages=[{"role": "user", "content": f"Generate content for {prompt} here is a random number {random.randint(0, 1000)}"}],
        stream=False
    )
    lines = resp.split("\n")
    end = []
    for line in lines:
        if len(line) > 0 and line[0].isdigit():
            end.append(line)
    resp = ' '.join(end)
    with open(f"{title}/script.txt", "w") as f:
        f.write(resp)

def auto():
    title = f"{len(os.listdir("./"))}-facts"
    os.makedirs(title, exist_ok=True)
    script(title, "5 random facts")
    render(title)
    auto()

if __name__ == '__main__':
    print("Welcome to TiktokGenerator")
    mode = selectMode()
    if mode == "script":
        title = input('Title ? ')
        os.makedirs(title, exist_ok=True)
        prompt = input('Prompt ? ')
        script(prompt)
    elif mode == "render":
        title = input('Title ? ')
        os.makedirs(title, exist_ok=True)
        render(title)
    elif mode == "auto":
        auto()