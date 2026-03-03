\# DVC Pipeline for Phoneme ASR Robustness to Noise



Evaluates robustness of `facebook/wav2vec2-lv-60-espeak-cv-ft` to additive 

noise across one or more languages, using a fully reproducible DVC pipeline.



\## Prerequisites



\- Python dependencies: `pip install -r requirements.txt`

\- \[espeak-ng](https://github.com/espeak-ng/espeak-ng/releases) installed on your system

\- On Windows, set: `PHONEMIZER\_ESPEAK\_LIBRARY=C:\\Program Files\\eSpeak NG\\libespeak-ng.dll`

\- LibriSpeech test-clean extracted to `data/raw/en/LibriSpeech/test-clean/`



\## Running

```

dvc repro

```



\## Adding a language



Add the language code to `languages` in `params.yaml` and re-run `dvc repro`. 

No code changes needed.

