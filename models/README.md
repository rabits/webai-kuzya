# Models Folder

This folder contains all runtime assets that are too large or too local to keep in git:

- F5-TTS checkpoint files
- Voice profiles with reference audio and transcripts
- Whisper checkpoints downloaded by the STT service

## Expected Layout

```text
models/
├── README.md
├── f5tts-russian/
│   ├── model_last_inference.safetensors
│   └── vocab.txt
├── whisper/
│   └── (downloaded by the stt container on first start)
└── voices/
    ├── default/
    │   ├── ref.wav
    │   └── ref.txt
    ├── female_1/
    │   ├── ref.wav
    │   └── ref.txt
    └── ...
```

## F5-TTS

Source:

- [Misha24-10/F5-TTS_RUSSIAN](https://huggingface.co/Misha24-10/F5-TTS_RUSSIAN/tree/main/F5TTS_v1_Base_accent_tune)

Put these files into `models/f5tts-russian/`:

- `model_last_inference.safetensors`
- `vocab.txt`

The service uses them through:

```ini
F5_MODEL_PATH=/app/models/f5tts-russian/model_last_inference.safetensors
F5_VOCAB_PATH=/app/models/f5tts-russian/vocab.txt
```

## Whisper

The STT container downloads the checkpoint named by `WHISPER_MODEL` into `models/whisper/` on first start. Default is `turbo` (Whisper large-v3-turbo), about 1.6 GB.

To pin a local copy in advance, start the stack once or place the official openai-whisper files in `models/whisper/` yourself.

## Voices

Each voice is a separate subfolder inside `models/voices/`.

Minimum required profile:

```text
models/voices/default/
├── ref.wav
└── ref.txt
```

Rules:

- `default/` must exist, because it is used as the startup fallback and warmup voice
- `ref.wav` should be a clean single-speaker reference sample
- `ref.txt` must contain the exact transcript of `ref.wav`

Recommended audio format:

- WAV
- 24 kHz
- mono
- 5 to 12 seconds

Example conversion:

```bash
ffmpeg -i input.mp3 -t 10 -ar 24000 -ac 1 -sample_fmt s16 models/voices/default/ref.wav
```

Example transcript creation:

```bash
echo "Текст того, что говорится в записи." > models/voices/default/ref.txt
```

To add another voice, create another folder:

```text
models/voices/female_1/
├── ref.wav
└── ref.txt
```

The TTS API lists voices at `GET /v1/audio/voices`. Open WebUI sends the folder name as the `voice` field on `POST /v1/audio/speech`.
