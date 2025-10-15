# SemanticSynth

A promptable synthesizer powered by [GloVe](https://nlp.stanford.edu/projects/glove/).


### To Run
- `python synth_training.py`
    - Ensure that audio is working properly
    - If submitting multiple words, separate your words with `,`, `.`, `;`, or a space.
    - THERE IS NO SPELL CHECK. Words not in the vector set will be ignored
- `python semantic_association.py`
    - Trains the model
- `python synth_promptable.py`
    - Loads the model for prediction

### Troubleshooting
- If you get a `sounddevice.PortAudioError` error:
    - Uninstall `portaudio` and `sounddevice`.
    - Install `portaudio` *first*.
    - Install `sounddevice`.
- If you get an