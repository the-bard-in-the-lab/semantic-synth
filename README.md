# SemanticSynth

A promptable synthesizer powered by [GloVe](https://nlp.stanford.edu/projects/glove/).


### To Run
- `interface.py` is the bulk of the application. a synth interface that lets you build your own corpus of sounds. It also lets you predict from the model you've already trained.
- The files in `text_interfaces` are the CLI predecessors to `interface.py`. They are mostly deprecated. You will need to move them up a level to run them.
    - `synth_training.py` is a text interface for training using randomized synth parameters
        - THERE IS NO SPELL CHECK. Words not in the vector set will be ignored.
        - Submit ONLY ONE WORD.
    - `synth_promptable.py` ;oads the model for prediction. This is more or less obsolete and has been replaced by `interface_kivy.py`.
- `semantic_association.py` trains the model.


### Troubleshooting
- Ensure that audio is working properly.
- If you get a `sounddevice.PortAudioError` error:
    - Uninstall `portaudio` and `sounddevice`.
    - Install `portaudio` *first*.
    - Install `sounddevice`.