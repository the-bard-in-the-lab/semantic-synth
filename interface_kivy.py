import tensorflow as tf
import numpy as np
import vector.vmath as vmath

import sounddevice as sd
import yaml
import synthesizer.synthesizer_4osc as synth
import synthesizer.melodies as melodies
import json

import os
os.environ["KIVY_NO_CONSOLELOG"] = "1"
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

config = yaml.safe_load(open("config.yaml"))
sr = config["audio_settings"]["sample_rate"]
filepath = config["corpus_file_path"]
#model = tf.keras.models.load_model(config["model_file_path"])
model = tf.keras.models.load_model("model_DECENT2.keras")
words, vectors = vmath.get_vector_data()

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
        
    def update_synth_from_widgets(self):
        self.my_synth.osc_1_mix = self.root.ids["sin"].value
        self.my_synth.osc_2_mix = self.root.ids["sq"].value
        self.my_synth.osc_2_sqpct = self.root.ids["sqpct"].value
        self.my_synth.osc_3_mix = self.root.ids["saw"].value
        self.my_synth.osc_4_mix = self.root.ids["noise"].value
        self.my_synth.adsr[0] = self.root.ids["a"].value
        self.my_synth.adsr[1] = self.root.ids["d"].value
        self.my_synth.adsr[2] = self.root.ids["s"].value
        self.my_synth.adsr[3] = self.root.ids["r"].value
        self.my_synth.lfo_1_freq = self.root.ids["vib_rate"].value
        self.my_synth.lfo_1_depth = self.root.ids["vib_depth"].value
        self.my_synth.lfo_2_freq = self.root.ids["trem_rate"].value
        self.my_synth.lfo_2_depth = self.root.ids["trem_depth"].value
    
    def update_widgets_from_synth(self):
        self.root.ids["sin"].value = float(self.my_synth.osc_1_mix)
        self.root.ids["sq"].value = float(self.my_synth.osc_2_mix)
        self.root.ids["sqpct"].value = float(self.my_synth.osc_2_sqpct)
        self.root.ids["saw"].value = float(self.my_synth.osc_3_mix)
        self.root.ids["noise"].value = float(self.my_synth.osc_4_mix)
        self.root.ids["a"].value = float(self.my_synth.adsr[0])
        self.root.ids["d"].value = float(self.my_synth.adsr[1])
        self.root.ids["s"].value = float(self.my_synth.adsr[2])
        self.root.ids["r"].value = float(self.my_synth.adsr[3])
        self.root.ids["vib_rate"].value = float(self.my_synth.lfo_1_freq)
        self.root.ids["vib_depth"].value = float(self.my_synth.lfo_1_depth)
        self.root.ids["trem_rate"].value = float(self.my_synth.lfo_2_freq)
        self.root.ids["trem_depth"].value = float(self.my_synth.lfo_2_depth)

    def play_sound(self, params):
        self.update_synth(params)
        print(self.my_synth.as_json())
        my_sound = self.my_synth.render(melodies.space01_short)
        
        sd.play(my_sound, sr)
        # sd.wait()

    def save_sound(self, name, params):
        self.update_synth(params)
        print(self.my_synth.as_json())
        self.commit(name)
        

    def save_sound(self, name):
        self.update_synth_from_widgets()
        print(self.my_synth.as_json())
        self.commit(name)
    
    def commit(self, name):
        with open(filepath, 'a') as file1:
            file1.writelines(self.my_synth.as_json() + "|" + name.strip() + "\n")
            print(f"Saved sound as {name}!")

    def load_sound(self, name):
        with open(filepath, 'r') as file1:
            # Split file1 at |, search linearly for first instance
            data = file1.readlines()
            mydata = ""
            for row in data:
                split = row.strip().split("|")
                #print(split[1])
                if split[1] == name:
                    mydata = split[0]
                    #print("Found it!")
                    break
            if mydata == "":
                print(f"Could not locate a sound called {name}.")
                return

            print(f"Found sound {name}!")
            print(mydata)
            dict = json.loads(mydata)
            self.update_synth([dict["osc_1_mix"], dict["osc_2_mix"], dict["osc_2_sqpct"], dict["osc_3_mix"], dict["osc_4_mix"], dict["adsr"], dict["lfo_1_freq"], dict["lfo_1_depth"], dict["lfo_2_freq"], dict["lfo_2_depth"]])
            print(self.my_synth.as_json())
            my_sound = self.my_synth.render(melodies.space01_short)
            self.update_widgets_from_synth()
            
            sd.play(my_sound, sr)

    def predict_sound_neural(self, name):
        ind = -1
        try:
            ind = words.index(name)
        except:
            print(f"Word {name} does not appear in the vector data. Please try a different word.")
            return
        
        my_vector = vectors[ind]
        my_synth_params = model.predict(np.array([my_vector]))
        print(my_synth_params[0])
        self.my_synth.assign_params_from_array(my_synth_params[0])
        self.update_widgets_from_synth()
        my_sound = self.my_synth.render(melodies.space01_short)
        sd.play(my_sound, sr)

    def predict_sound_knn(self, name):
        # First, get the vectors associated with the saved synth configurations
        synths = []
        synthwords = []
        print("Loading synth data . . .")
        with open(filepath, 'r') as file:
            lines = file.readlines()
            for line in range(len(lines)):
                line_split = lines[line].split("|")
                synths.append(line_split[0])
                synthwords.append(line_split[1].strip())
        
        print("Getting word vectors for synth data (This needs to be optimized) . . .")
        word_vectors = vmath.get_vector_data_from_names(synthwords, words, vectors)
        
        # Check if vector in dataset. If yes, return configuration.
        print("Checking against corpus for exact match . . .")
        if name in synthwords:
            print(f"Found sound {name} in corpus!")

            dict = json.loads(synths[synthwords.index(name)])
            self.update_synth([dict["osc_1_mix"], dict["osc_2_mix"], dict["osc_2_sqpct"], dict["osc_3_mix"], dict["osc_4_mix"], dict["adsr"], dict["lfo_1_freq"], dict["lfo_1_depth"], dict["lfo_2_freq"], dict["lfo_2_depth"]])
            self.update_widgets_from_synth()
            my_sound = self.my_synth.render(melodies.space01_short)
            sd.play(my_sound, sr)
            return

        # Then, find the k nearest vectors AMONG THOSE SAVED CONFIGURATIONS
        print("Running KNN search . . .")
        print(f"Key: {name}")
        # print(f"Key vector: {vectors[words.index(name)]}")
        # print(f"Forest: {word_vectors}")
        print(f"k: {config["k"]}")
        neighbors = vmath.get_k_nearest_neighbors(vectors[words.index(name)], word_vectors, config["k"])
        neighbors_named = []
        print("Neighbors:")
        vlist = vectors.tolist()
        for wordvec in neighbors:
            print(words[vlist.index(wordvec.tolist())])
            neighbors_named.append(words[vlist.index(wordvec.tolist())])

        # Weighted average. Cosine distance means bigger is further away.
        # - Get 1/cosine distance for each neighbor
        # - Sum the reciprocal cosine distances
        # - Multiply each set of params by fraction of rcd sum
        # - Sum the params and return
        distances = [float(vmath.cosine_distance(vectors[words.index(name)], k)) for k in neighbors]
        inverses = [1/i for i in distances]
        total = sum(inverses)
        fracs = [i/total for i in inverses]
        
        param_dicts = [json.loads(synths[synthwords.index(neighbors_named[i])]) for i in range(len(neighbors_named))]
        
        # print(fracs)
        # print(sum(fracs))
        # print(param_dicts)
        # print(len(param_dicts))
        final_params = {}
        for i in range(len(param_dicts)):
            for key in param_dicts[i]:
                #print(i, key)
                if key == "adsr":
                    for j in range(4):
                        param_dicts[i][key][j] *= fracs[i]
                    if key in final_params:
                        for j in range(4):
                            final_params[key][j] += param_dicts[i][key][j]
                    else:
                        final_params[key] = param_dicts[i][key]
                else:    
                    param_dicts[i][key] *= fracs[i]
                    if key in final_params:
                        final_params[key] += param_dicts[i][key]
                    else:
                        final_params[key] = param_dicts[i][key]
        #print(final_params)
        dict = final_params
        self.update_synth([dict["osc_1_mix"], dict["osc_2_mix"], dict["osc_2_sqpct"], dict["osc_3_mix"], dict["osc_4_mix"], dict["adsr"], dict["lfo_1_freq"], dict["lfo_1_depth"], dict["lfo_2_freq"], dict["lfo_2_depth"]])
        self.update_widgets_from_synth()
        my_sound = self.my_synth.render(melodies.space01_short)
        sd.play(my_sound, sr)


    def predict_sound(self, name):
        #self.predict_sound_neural(name)
        self.predict_sound_knn(name)

class SemanticSynthInterface(BoxLayout):
    pass

if __name__ == '__main__':
    SemanticSynthApp().run()