import numpy as np
import sounddevice as sd
import yaml
import synthesizer.synthesizer_4osc as synth
import melodies
from concurrent.futures import ThreadPoolExecutor

config = yaml.safe_load(open("config.yaml"))
sr = config["audio_settings"]["sample_rate"]
filepath = config["corpus_file_path"]



def play_sound(param):
    my_synth = synth.synth(0, 1, 0.5, 0, 0,
                        adsr=[0.1, 0.1, 0.5, 2],
                        lfo_1=[2, 0],
                        lfo_2=[10, 0])
    
    # TODO: Replace this with parameters from interface
    my_synth.osc_2_sqpct = param / 100.0
    my_sound = my_synth.render(melodies.space01)
    #print(my_synth.as_json())
    print("Square percent:", param)
    sd.play(my_sound, sr)
    # sd.wait()