from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix import boxlayout

import numpy as np
import sounddevice as sd
import yaml
import synthesizer.synthesizer_4osc as synth
import melodies
from concurrent.futures import ThreadPoolExecutor

config = yaml.safe_load(open("config.yaml"))
sr = config["audio_settings"]["sample_rate"]
filepath = config["corpus_file_path"]



class SemanticSynthApp(App):
    # (See .kv file)

    my_synth = synth.synth(0, 1, 0.5, 0, 0,
                        adsr=[0.1, 0.1, 0.5, 2],
                        lfo_1=[2, 0],
                        lfo_2=[10, 0])

    def play_sound(self, params):
        '''
        The parameters are:
        00 Sine mix
        01 Square mix
        02 Square percent (as an integer 0-100)
        03 Saw mix
        04 Noise mix
        05 ADSR as an array[A,D,S,R]
        06 Tremolo frequency
        07 Tremolo depth
        08 Vibrato frequency
        09 Vibrato depth
        '''

        self.my_synth.osc_1_mix = params[0]
        self.my_synth.osc_2_mix = params[1]
        self.my_synth.osc_2_sqpct = params[2]
        self.my_synth.osc_3_mix = params[3]
        self.my_synth.osc_4_mix = params[4]
        self.my_synth.adsr = params[5]
        self.my_synth.lfo_2_freq = params[6]
        self.my_synth.lfo_2_depth = params[7]
        self.my_synth.lfo_1_freq = params[8]
        self.my_synth.lfo_1_depth = params[9]

        print(self.my_synth.as_json())
        my_sound = self.my_synth.render(melodies.space01_short)
        
        sd.play(my_sound, sr)
        # sd.wait()

    def save_sound(self, name):
        with open(filepath, 'a') as file1:
            file1.writelines(self.my_synth.as_json() + "|" + name.strip() + "\n")
            print("Saved!")

    def load_sound(self, name):
        # Holy hell this one is going to suck
        print("Loading sounds is currently under construction. Please check back later!")

if __name__ == '__main__':
    SemanticSynthApp().run()