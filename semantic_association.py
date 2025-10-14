import numpy as np
import yaml
import tensorflow as tf
import os
import json
import re
from vector.vmath import get_vector_data
from vector.vmath import get_k_nearest_neighbors


config = yaml.safe_load(open("config.yaml"))
k = config["k"]

type_dict = {"SINE":0, "SQUARE":1, "SAW":2, "NOISE":3}

def create_model(input_dim: int=50, output_dim: int=14):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dropout(0.05),
        tf.keras.layers.Dense(output_dim)
    ])
    return model

def load_training_data(vwords : np.array, vvectors : np.array):
    with open(config["corpus_file_path"], 'r') as file:
        lines = file.readlines()
        synths = []
        words = []
        for line in range(len(lines)):
            line_split = lines[line].split("|")
            synth = json.loads(line_split[0])
            #synth_arr = np.zeros([19,])
            synth_arr = np.zeros([14,])

            # Assign parameter values:
            # synth_arr[type_dict[synth["osc_1"]]] = 1
            # synth_arr[4] = synth["osc_1_sqpct"]
            # synth_arr[type_dict[synth["osc_2"]] + 5] = 1
            # synth_arr[9] = synth["osc_2_sqpct"]
            # synth_arr[10] = synth["mix_pct"]
            # synth_arr[11:15] = synth["adsr"][0:4]
            # synth_arr[15] = synth["lfo_1_freq"]
            # synth_arr[16] = synth["lfo_1_depth"]
            # synth_arr[17] = synth["lfo_2_freq"]
            # synth_arr[18] = synth["lfo_2_depth"]

            synth_arr[0] = synth["osc_1_mix"]
            synth_arr[1] = synth["osc_2_mix"]
            synth_arr[2] = synth["osc_3_mix"]
            synth_arr[3] = synth["osc_4_mix"]
            synth_arr[4] = synth["osc_2_sqpct"]
            synth_arr[5:9] = synth["adsr"][0:4]
            synth_arr[10] = synth["lfo_1_freq"]
            synth_arr[11] = synth["lfo_1_depth"]
            synth_arr[12] = synth["lfo_2_freq"]
            synth_arr[13] = synth["lfo_2_depth"]

            my_words = re.split(r"[,.;\s]", line_split[1].strip())
            my_words = [x for x in my_words if x != ""]
            #vlist = vvectors.tolist()
            #print(my_words)
            for word in my_words:
                try:
                    ind = vwords.index(word)
                    #print(ind)
                    
                    #words.append(vvectors[ind])
                    #synths.append(synth_arr)
                    
                    # DATASET AUGMENTATION:
                    # We find the k nearest words in the dataset and
                    # add them to the training data
                    # Note that k includes the word itself
                    neighbors = get_k_nearest_neighbors(vvectors[ind], vvectors, k)
                    #print("- - - - -")
                    for wordvector in neighbors:
                        words.append(wordvector)
                        synths.append(synth_arr)
                        #print(vwords[vlist.index(wordvector.tolist())])

                except:
                    print(f"Word {word} not found in vector corpus; skipping . . .")
        print("LENGTHS")
        print(np.array(synths).shape)
        print(np.array(words).shape)

        print(synths[0])
        print(words[0])
        return np.array(words), np.array(synths)

def main():
    words, vectors = get_vector_data()
    #print(words)
    print(vectors)


    # Tensorflow model setup
    model = create_model()
    loss_fn = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
    
    if (os.path.isfile(config["weights_file_path"])):
        model.load_weights(config["weights_file_path"])

    training_data, training_labels = load_training_data(words, vectors)
    model.fit(training_data, training_labels, epochs=15)

    model.save_weights(config["weights_file_path"])
    model.save(config["model_file_path"])


    

if __name__ == "__main__":
    main()