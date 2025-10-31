import numpy as np
import sounddevice as sd
import yaml
import synthesizer.synthesizer_4osc as synth
import melodies

config = yaml.safe_load(open("config.yaml"))
sr = config["audio_settings"]["sample_rate"]
filepath = config["corpus_file_path"]

def main():
    # my_modulator = sine(2, 3.0, 1)
    # my_sound = fm_synthesis(220, my_modulator, WaveType.SQUARE)
    # sd.play(my_sound, config["audio_settings"]["sample_rate"], [2])
    # sd.wait()
    
    my_synth = synth.synth(1, 0, 0.5, 0, 0,
                            adsr=[0.1, 0.1, 0.5, 2],
                            lfo_1=[2, 0],
                            lfo_2=[10, 0])
    
    # Build the model
    while True:
        my_synth.randomize_params()
        my_sound = my_synth.render(melodies.space01)
        #print(my_synth.as_json())
        sd.play(my_sound, sr)
        sd.wait()
        descriptors = input("Describe the sound you just heard: ")
        with open(filepath, 'a') as file1:
            file1.writelines(my_synth.as_json() + "|" + descriptors + "\n")
    

if __name__ == "__main__":
    main()