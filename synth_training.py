import numpy as np
import sounddevice as sd
import yaml
import synthesizer.synthesizer_4osc as synth

config = yaml.safe_load(open("config.yaml"))
sr = config["audio_settings"]["sample_rate"]
filepath = config["corpus_file_path"]

notes = np.array([[60, .8],
                  [67, .8],
                  [65, .2],
                  [64, .2],
                  [62, .2],
                  [64, .2],
                  [60, .8],
                  [55, 1.6],
                  [57, .8],
                  [69, .8],
                  [67, .2],
                  [66, .2],
                  [64, .2],
                  [66, .2],
                  [62, .8],
                  [71, 1.6],
                  ])

def main():
    # my_modulator = sine(2, 3.0, 1)
    # my_sound = fm_synthesis(220, my_modulator, WaveType.SQUARE)
    # sd.play(my_sound, config["audio_settings"]["sample_rate"], [2])
    # sd.wait()

    # my_sound = sine(3, 3)
    # fm_sound = fm_synthesis(220, my_sound)
    # sd.play(fm_sound, config["audio_settings"]["sample_rate"])
    # sd.wait() 

    # my_synth = synth.synth(synth.WaveType.SAW, 0.1, synth.WaveType.SINE, 0.5,
    #                  mix_pct=0.9,
    #                  adsr=[0.1, 0.1, 0.5, 2],
    #                  lfo_1=[2, 0],
    #                  lfo_2=[10, 0])

    my_synth = synth.synth(1, 0, 0.5, 0, 0,
                            adsr=[0.1, 0.1, 0.5, 2],
                            lfo_1=[2, 0],
                            lfo_2=[10, 0])
    
    # Build the model
    while True:
        my_synth.randomize_params()
        my_sound = my_synth.render(notes)
        #print(my_synth.as_json())
        sd.play(my_sound, sr)
        sd.wait()
        descriptors = input("Describe the sound you just heard: ")
        with open(filepath, 'a') as file1:
            file1.writelines(my_synth.as_json() + "|" + descriptors + "\n")
    

if __name__ == "__main__":
    main()