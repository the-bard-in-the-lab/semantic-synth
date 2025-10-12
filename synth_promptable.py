import numpy as np
import sounddevice as sd
import yaml
import tensorflow as tf
import synthesizer.synthesizer_4osc as synth
import os

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

# def create_model(input_dim: int=50, output_dim: int=19):
#     model = tf.keras.models.Sequential([
#         tf.keras.layers.Input(shape=(input_dim,)),
#         tf.keras.layers.Dense(50, activation='relu'),
#         tf.keras.layers.Dense(25, activation='relu'),
#         tf.keras.layers.Dropout(0.3),
#         tf.keras.layers.Dense(output_dim)
#     ])
#     return model

def create_model(input_dim: int=100, output_dim: int=14):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(25, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(output_dim)
    ])
    return model

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

def main():
    # my_synth = synth.synth(synth.WaveType.SAW, 0.1, synth.WaveType.SINE, 0.5,
    #                  mix_pct=0.9,
    #                  adsr=[0.1, 0.1, 0.5, 2],
    #                  lfo_1=[2, 0],
    #                  lfo_2=[10, 0]) # Sample parameters

    my_synth = synth.synth(1, 0, 0.5, 0, 0,
                            adsr=[0.1, 0.1, 0.5, 2],
                            lfo_1=[2, 0],
                            lfo_2=[10, 0])
    
    # Load the model
    model = create_model()
    loss_fn = tf.keras.losses.MeanSquaredError()
    model.compile(#optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
    
    if (os.path.isfile(config["weights_file_path"])):
        model.load_weights(config["weights_file_path"])
    else:
        print(f"Could not find model at {config["weights_file_path"]}")
        quit()

    words, vectors = get_vector_data()

    while True:
        descriptor = input("In a word, describe the synth tone you are looking for: ")
        ind = -1
        try:
            ind = words.index(descriptor)
        except:
            print(f"Word {descriptor} does not appear in the vector data. Please try a different word.")
            continue
        
        my_vector = vectors[ind]
        print(my_vector)
        print(my_vector.shape)
        my_synth_params = model.predict(np.array([my_vector]))
        print(my_synth_params[0])
        my_synth.assign_params_from_array(my_synth_params[0])
        my_sound = my_synth.render(notes)
        #print(my_synth.as_json())
        sd.play(my_sound, sr)
        sd.wait()
    
if __name__ == "__main__":
    main()