import os
import re
import argparse
import subprocess
import whisper

def main(media_folder, whisper_model, source_language, target_language, prompt, task):
    # Audio and video files with specified extensions will be processed
    audio_exts = ['mp3', 'aac', 'ogg', 'wav']
    video_exts = ['mp4', 'avi', 'mov', 'mkv']

    print(f'Looking into "{media_folder}"')
    files = [file for file in os.listdir(media_folder) if match_ext(file, audio_exts + video_exts)]
    print(f'Found {len(files)} files:')
    for filename in files: print(filename)
    for filename in files:
        print(f'\n\nProcessing {filename}')
        media_file = os.path.join(media_folder, filename)
        if filename.split('.')[-1] in video_exts:
            audio_file = extract_audio(media_file)
            process_audiofile(audio_file, whisper_model, media_file, source_language, target_language, prompt, task)
            os.remove(audio_file)  # Clean up the temporary audio file
        else:
            process_audiofile(media_file, whisper_model, source_language, target_language, prompt, task)

def match_ext(filename, extensions):
    return filename.split('.')[-1] in extensions

def extract_audio(video_file):
    audio_file = video_file.rsplit('.', 1)[0] + '.wav'
    command = [
        'ffmpeg',
        '-i', video_file,
        '-q:a', '0',
        '-map', 'a',
        audio_file
    ]
    subprocess.run(command, check=True)
    return audio_file

def process_audiofile(fname, whisper_model, original_file=None, source_language='auto', target_language=None, prompt=None, task='transcribe'):
    fext = fname.split('.')[-1]
    fname_noext = fname[:-(len(fext)+1)]

    model = whisper.load_model(whisper_model)
    
    if source_language == 'auto':
        # Detect the spoken language
        audio = whisper.load_audio(fname)
        audio = whisper.pad_or_trim(audio)

        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        _, probs = model.detect_language(mel)
        source_language = max(probs, key=probs.get)
        #print(f"Detected language: {source_language}")

    if task == 'translate':
        if not target_language:
            raise ValueError("Target language must be specified for translation")
        # For translation, pass the target language directly in `transcribe`
        result = model.transcribe(fname, verbose=True, language=source_language, prompt=prompt, task=task, target_language=target_language)
    else:
        # For transcription
        result = model.transcribe(fname, verbose=True, language=source_language, prompt=prompt)

    # Create timecode file
    with open(fname_noext + '_timecode'+whisper_model+'.txt', 'w', encoding='UTF-8') as f:
        for segment in result['segments']:
            timecode_sec = int(segment['start'])
            hh = timecode_sec // 3600
            mm = (timecode_sec % 3600) // 60
            ss = timecode_sec % 60
            timecode = f'[{str(hh).zfill(2)}:{str(mm).zfill(2)}:{str(ss).zfill(2)}]'
            text = segment['text']
            f.write(f'{timecode} {text}\n')

    # Create raw text file
    rawtext = ' '.join([segment['text'].strip() for segment in result['segments']])
    rawtext = re.sub(" +", " ", rawtext)

    with open(fname_noext + '.txt', 'w', encoding='UTF-8') as f:
        f.write(rawtext)

    # Create SRT file
    if original_file:
        srt_file = fname_noext + '.srt'
        with open(srt_file, 'w', encoding='UTF-8') as f:
            for idx, segment in enumerate(result['segments']):
                start_time = format_time(segment['start'])
                end_time = format_time(segment['end'])
                text = segment['text']
                f.write(f"{idx + 1}\n{start_time} --> {end_time}\n{text}\n\n")

def format_time(seconds):
    hh = int(seconds) // 3600
    mm = (int(seconds) % 3600) // 60
    ss = int(seconds) % 60
    ms = int((seconds - int(seconds)) * 1000)
    return f"{hh:02}:{mm:02}:{ss:02},{ms:03}"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process audio and video files for transcription.')
    parser.add_argument('media_folder', type=str, help='Folder containing media files to process')
    parser.add_argument('whisper_model', type=str, help='Whisper model to use for transcription')
    parser.add_argument('--source_language', type=str, default='auto', help='Source language code for transcription or translation (default: auto)')
    parser.add_argument('--target_language', type=str, default=None, help='Target language code for translation (required if task is "translate")')
    parser.add_argument('--prompt', type=str, default=None, help='Prompt to guide the transcription')
    parser.add_argument('--task', type=str, choices=['transcribe', 'translate'], default='transcribe', help='Task to perform (default: transcribe)')
    
    args = parser.parse_args()

    if args.task == 'translate' and not args.target_language:
        parser.error('--target_language is required when task is "translate"')

    # Calling main() function with media folder, whisper model, and additional arguments
    main(args.media_folder, args.whisper_model, args.source_language, args.target_language, args.prompt, args.task)