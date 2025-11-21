# SemanticSynth

A promptable synthesizer powered by [GloVe](https://nlp.stanford.edu/projects/glove/).


### To Run
- `interface_kivy.py` is a synth interface that lets you build your own corpus of sounds. It also lets you predict from the model you've already trained.
- `python synth_training.py` is a text interface for training using randomized synth parameters
    - THERE IS NO SPELL CHECK. Words not in the vector set will be ignored.
    - Submit ONLY ONE WORD.
- `python semantic_association.py`
    - Trains the model
- `python synth_promptable.py`
    - Loads the model for prediction. This is more or less obsolete and has been replaced by `interface_kivy.py`.

### Troubleshooting
- Ensure that audio is working properly.
- If you get a `sounddevice.PortAudioError` error:
    - Uninstall `portaudio` and `sounddevice`.
    - Install `portaudio` *first*.
    - Install `sounddevice`.