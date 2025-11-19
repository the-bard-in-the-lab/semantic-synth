from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout

import numpy as np
import sounddevice as sd
import yaml
import synthesizer.synthesizer_4osc as synth
import melodies
import json
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

    def build(self):
        return SemanticSynthInterface()

    def update_synth(self, params):
        '''
        The parameters are:
        00 Sine mix
        01 Square mix
        02 Square percent
        03 Saw mix
        04 Noise mix
        05 ADSR as an array[A,D,S,R]
        06 Vibrato frequency
        07 Vibrato depth
        08 Tremolo frequency
        09 Tremolo depth
        
        '''

        self.my_synth.osc_1_mix = params[0]
        self.my_synth.osc_2_mix = params[1]
        self.my_synth.osc_2_sqpct = params[2]
        self.my_synth.osc_3_mix = params[3]
        self.my_synth.osc_4_mix = params[4]
        self.my_synth.adsr = params[5]
        self.my_synth.lfo_1_freq = params[6]
        self.my_synth.lfo_1_depth = params[7]
        self.my_synth.lfo_2_freq = params[8]
        self.my_synth.lfo_2_depth = params[9]
        

    def play_sound(self, params):
        self.update_synth(params)
        print(self.my_synth.as_json())
        my_sound = self.my_synth.render(melodies.space01_short)
        
        sd.play(my_sound, sr)
        # sd.wait()

    def save_sound(self, name, params):
        self.update_synth(params)
        print(self.my_synth.as_json())
        print(self.root.ids["sound_name"].text)
        self.root.ids["sound_name"].text = ""
        
        with open(filepath, 'a') as file1:
            file1.writelines(self.my_synth.as_json() + "|" + name.strip() + "\n")
            print(f"Saved sound as {name}!")
            

    def load_sound(self, name):
        with open(filepath, 'r') as file1:
            # file1.writelines(self.my_synth.as_json() + "|" + name.strip() + "\n")
            
            # Split file1 at |, search linearly for first instance
            data = file1.readlines()
            mydata = ""
            for row in data:
                split = row.split("|")
                if split[1] == name:
                    mydata = split[0]
            if mydata == "":
                print(f"Could not locate a sound called {name}.")
                return

            print(f"Found sound {name}!")
            print(mydata)
            dict = json.loads(mydata)
            self.update_synth([dict["osc_1_mix"], dict["osc_2_mix"], dict["osc_2_sqpct"], dict["osc_3_mix"], dict["osc_4_mix"], dict["adsr"], dict["lfo_1_freq"], dict["lfo_1_depth"], dict["lfo_2_freq"], dict["lfo_2_depth"]])
            print(self.my_synth.as_json())
            my_sound = self.my_synth.render(melodies.space01_short)
            
            sd.play(my_sound, sr)
            # Assign the values
            self.root.ids["sin"].value = self.my_synth.osc_1_mix
            self.root.ids["sq"].value = self.my_synth.osc_2_mix
            self.root.ids["sqpct"].value = self.my_synth.osc_2_sqpct
            self.root.ids["saw"].value = self.my_synth.osc_3_mix
            self.root.ids["noise"].value = self.my_synth.osc_4_mix
            self.root.ids["a"].value = self.my_synth.adsr[0]
            self.root.ids["d"].value = self.my_synth.adsr[1]
            self.root.ids["s"].value = self.my_synth.adsr[2]
            self.root.ids["r"].value = self.my_synth.adsr[3]
            self.root.ids["vib_rate"].value = self.my_synth.lfo_1_freq
            self.root.ids["vib_depth"].value = self.my_synth.lfo_1_depth
            self.root.ids["trem_rate"].value = self.my_synth.lfo_2_freq
            self.root.ids["trem_depth"].value = self.my_synth.lfo_2_depth

class SemanticSynthInterface(BoxLayout):
    pass

if __name__ == '__main__':
    SemanticSynthApp().run()