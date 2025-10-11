import numpy as np
import yaml
import tensorflow as tf
import os
import json
import re


config = yaml.safe_load(open("config.yaml"))

type_dict = {"SINE":0, "SQUARE":1, "SAW":2, "NOISE":3}

def get_vector_data():
    with open(config["vector_file_path"], 'r') as file:
        lines = file.readlines()
        vectors = np.empty([len(lines), len(lines[0].split()) - 1])
        words = []
        for line in range(len(lines)):#range(10):
            thisline = lines[line].split()
            word = thisline[0]
            words.append(word)
            #print(word)
            del thisline[0]
            thisline = [float(i) for i in thisline]
            vector = np.array(thisline)
            #print(vector)
            #words = np.append(words, word)
            vectors[line] = vector            

        return words, vectors

def create_model(input_dim: int=50, output_dim: int=19):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(25, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(output_dim)
    ])
    return model

def load_training_data(vwords, vvectors):
    with open(config["corpus_file_path"], 'r') as file:
        lines = file.readlines()
        synths = []
        words = []
        for line in range(len(lines)):
            line_split = lines[line].split("|")
            synth = json.loads(line_split[0])
            synth_arr = np.zeros([19,])

            # Assign parameter values:
            synth_arr[type_dict[synth["osc_1"]]] = 1
            synth_arr[4] = synth["osc_1_sqpct"]
            synth_arr[type_dict[synth["osc_2"]] + 5] = 1
            synth_arr[9] = synth["osc_2_sqpct"]
            synth_arr[10] = synth["mix_pct"]
            synth_arr[11:15] = synth["adsr"][0:4]
            synth_arr[15] = synth["lfo_1_freq"]
            synth_arr[16] = synth["lfo_1_depth"]
            synth_arr[17] = synth["lfo_2_freq"]
            synth_arr[18] = synth["lfo_2_depth"]

            my_words = re.split(r"[,.;\s]", line_split[1].strip())
            my_words = [x for x in my_words if x != ""]
            #print(my_words)
            for word in my_words:
                try:
                    ind = vwords.index(word)
                    print(ind)
                    words.append(vvectors[ind])
                    synths.append(synth_arr)
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
    print(words)
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
    model.fit(training_data, training_labels, epochs=3)

    model.save_weights(config["weights_file_path"])


    

if __name__ == "__main__":
    main()