#TODO: Implement nearest neighbors function
import numpy as np
from scipy.spatial import distance
import yaml

config = yaml.safe_load(open("config.yaml"))

def get_vector_data():
    with open(config["vector_file_path"], 'r') as file:
        lines = file.readlines()
        vectors = np.empty([len(lines), len(lines[0].split()) - 1])
        words = []
        defaultlength = len(lines[0].split())
        for line in range(len(lines)):#range(10):
            thisline = lines[line].strip().split()
            if len(thisline) != defaultlength:
                # Occasionally there are weird words with spaces because WHY, STANFORD NLP?!?!
                # WHY WOULD YOU PUT SPACES IN THE MIDDLE OF WORDS?!?!?!
                words.append("NULL")
                continue
            word = thisline[0]
            words.append(word)
            
            del thisline[0]
            thisline = [float(i) for i in thisline]
            vector = np.array(thisline)
            #print(vector)
            #words = np.append(words, word)
            vectors[line] = vector            

        return words, vectors

def get_k_nearest_neighbors(key: np.array, forest: np.array, k: int):
    # Key is the vector we're looking for
    # Forest is the array of vectors we're searching through

    # Debug:
    # print(f"Key shape: {np.shape(key)}")
    # print(f"Forest shape: {np.shape(forest)}")

    # Compute cosine distance
    #dist = np.apply_along_axis(lambda x : abs(distance.cosine(x, key)), 1, forest)
    dist = 1 - (key.dot(forest.T) / np.linalg.norm(forest, axis=1)) 

    # We can be sneaky and skip including the key norm because it's
    # a constant. We don't care about the actual values, only the order.

    # Partition by size, then take the first k only
    inds = np.argpartition(dist, k)[:k]
    
    # List comprehension because it's fun :)
    return [forest[j] for j in inds]

if __name__ == "__main__":
    # Demo
    print("This is a demo of the partion function.")

    print("Let's find the closest words to 'frog' in the dataset (including 'frog' itself).")

    print("Importing data . . .")
    words, vectors = get_vector_data()
    test_word = "frog"
    k = 5
    print(f"About to get k nearest neighbors (k={k})")
    neighbors = get_k_nearest_neighbors(vectors[words.index(test_word)], vectors, k)
    
    print(f"Test word: {test_word}")
    print(f"k: {k}")
    print("Neighbors:")
    vlist = vectors.tolist()
    for wordvec in neighbors:
        print(words[vlist.index(wordvec.tolist())])
        
    #Plurals demo
    test_word2_sing = "cat"
    test_word2_pl = "cats"
    
    diff = vectors[words.index(test_word2_pl)] - vectors[words.index(test_word2_sing)]

    test_word3 = "dog"
    test_word4 = vectors[words.index(test_word3)] + diff

    neighbors = get_k_nearest_neighbors(test_word4, vectors, 5)
    
    print(f"Test word: {test_word3}")
    print(f"diff: {diff}")
    print("Neighbors:")
    vlist = vectors.tolist()
    for wordvec in neighbors:
        print(words[vlist.index(wordvec.tolist())])