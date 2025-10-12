import numpy as np
import sounddevice as sd
import yaml
from enum import Enum
import json

config = yaml.safe_load(open("config.yaml"))
sr = config["audio_settings"]["sample_rate"]
filepath = config["corpus_file_path"]


# TODO: make WaveType an int enum and remove type_dict_reverse
class WaveType(int, Enum):
    SINE = 0
    SQUARE = 1
    SAW = 2
    NOISE = 3

def sine(freq: float=440.0, duration: float=1.0, amplitude: float=0.5):
    n_samples = int(sr * duration)
    time_points = np.linspace(0, duration, n_samples, False)
    sine = np.sin(2 * np.pi * freq * time_points)
    sine *= amplitude
    return sine

def square(freq: float=440.0, duration: float=1.0, amplitude: float=0.5, pct: float=0.5):
    sr = config["audio_settings"]["sample_rate"]
    if pct > 1 or pct < 0:
        raise Exception("Square pct " + pct + " out of range")

    n_samples = int(sr * duration)
    square = np.ones(n_samples)
    
    cycle_length = sr / freq #(NB: cycle is a float!!!)
    for i in range(n_samples):
        if (i % cycle_length) / cycle_length > pct:
            square[i] = -1

    square *= amplitude
    return square

def saw(freq: float=440.0, duration: float=1.0, amplitude: float=0.5):
    n_samples = int(sr * duration)
    time_points = np.linspace(0, duration, n_samples, False)
    saw = saw_function(2 * np.pi * freq * time_points)
    saw *= amplitude
    return saw

def noise(duration: float=1.0, amplitude: float=0.5):
    n_samples = int(sr * duration)
    noise = 2 * np.random.rand(n_samples) - 1
    noise *= amplitude
    return noise

def saw_function(deg: float):
    saw = (deg / (2 * np.pi)) % 1 # goes from 0 to 1
    return 2 * saw - 1 # goes from -1 to 1

def square_function(deg: float, square_pct: float=0.5):
    pct = deg / (2 * np.pi) % 1
    return 1.0 if pct > square_pct else -1.0

def square_function(deg: np.array, square_pct: float=0.5):
    pct = deg / (2 * np.pi) % 1
    return np.where(pct > square_pct, -1, 1)
    
def noise_function(deg: float):
    return 2 * np.random.random() - 1

def apply_envelope(sound: np.array, adsr: list) -> np.array:
    
    sound = sound.copy() # Python objects are pass by reference

    attack_samples = int(adsr[0] * sr)
    decay_samples = int(adsr[1] * sr)
    release_samples = int(adsr[3] * sr)
    if (attack_samples + decay_samples + release_samples > len(sound)):
        decay_samples -= attack_samples + decay_samples + release_samples - len(sound)
    sustain_samples = len(sound) - (attack_samples + decay_samples + release_samples)

    sound[:attack_samples] *= np.linspace(0, 1, attack_samples)
    sound[attack_samples:attack_samples + decay_samples] *= np.linspace(1, adsr[2], decay_samples)
    sound[attack_samples + decay_samples:attack_samples + decay_samples+sustain_samples] *= adsr[2]
    sound[attack_samples + decay_samples+sustain_samples:] *= np.linspace(adsr[2], 0, release_samples)

    return sound

def am_synthesis(carrier_freq: float, modulator_wave: np.array, modulation_index: float=0.5, amplitude: float=0.5):
    total_samples = len(modulator_wave)

    time_points = np.arange(total_samples) / sr

    carrier_wave = np.sin(2 * np.pi * carrier_freq * time_points)

    am_wave = (1 + modulation_index * modulator_wave) * carrier_wave
    max_amplitude = np.max(np.abs(am_wave))
    am_wave = amplitude * (am_wave / max_amplitude)
    return am_wave

def am_synthesis(carrier_wave: np.array, modulator_wave: np.array, modulation_index: float=0.5, amplitude: float=0.5):
    am_wave = (1 + modulation_index * modulator_wave) * carrier_wave
    max_amplitude = np.max(np.abs(am_wave))
    am_wave = amplitude * (am_wave / max_amplitude)
    return am_wave

def fm_synthesis(carrier_freq: float, modulator_wave: np.array, wave_type: WaveType, modulation_index: float=5, amplitude: float=0.5, square_pct: float=0.5):
    total_samples = len(modulator_wave)

    time_points = np.arange(total_samples) / sr


    #fm_wave = np.sin(2 * np.pi * carrier_freq * time_points + modulation_index * modulator_wave)
    fm_wave = []
    if wave_type == WaveType.SAW:
        fm_wave = saw_function(2 * np.pi * carrier_freq * time_points + modulation_index * modulator_wave)
    elif wave_type == WaveType.SQUARE:    
        fm_wave = square_function(2 * np.pi * carrier_freq * time_points + modulation_index * modulator_wave, square_pct)
    else:
        fm_wave = np.sin(2 * np.pi * carrier_freq * time_points + modulation_index * modulator_wave)
    

    max_amplitude = np.max(np.abs(fm_wave))
    fm_wave = amplitude * (fm_wave / max_amplitude)
    return fm_wave

def m_to_f(val: float, reference: float=440):
    return reference * 2 ** ((val - 69) / 12)

def f_to_m(val: float, reference: float=440):
    pass #Look up change of base

class synth:
    osc_1_mix = 0
    osc_2_mix = 0
    osc_2_sqpct = 0.5
    osc_3_mix = 0
    osc_4_mix = 0
    #adsr = [0.01, 0.1, 0.005, 1.0]
    adsr = [0.1, 0.5, 0.5, 1.0]
    lfo_1_freq = 2
    lfo_1_depth = 1
    lfo_2_freq = 2
    lfo_2_depth = 1
    def __init__(self, osc_1_mix: float=0, osc_2_mix: float=0, osc_2_sqpct: float=0.5, osc_3_mix: float=0, osc_4_mix: float=0, adsr: list=[0.1, 0.1, 0.5, 1.0], lfo_1: list = [2, 1], lfo_2: list = [2, 1]):
        self.osc_1_mix = osc_1_mix
        self.osc_2_mix = osc_2_mix
        self.osc_2_sqpct = osc_2_sqpct
        self.osc_3_mix = osc_3_mix
        self.osc_4_mix = osc_4_mix
        self.adsr = adsr
        self.lfo_1_freq = lfo_1[0]
        self.lfo_1_depth = lfo_1[1]
        self.lfo_2_freq = lfo_2[0]
        self.lfo_2_depth = lfo_2[1]
    
    def randomize_params(self):
        # pick two oscillators:
        osc1 = int(4 * np.random.random() ** config["osc_skew_factor"])
        osc2 = int(4 * np.random.random() ** config["osc_skew_factor"])
        osc1mix = 0
        osc2mix = 0
        if osc1 == osc2:
            osc1mix = 1
            osc2mix = 1
        else: # Most things should end up here
            osc1mix = np.random.random()
            osc2mix = 1 - osc1mix
        sqpct = np.random.random()
        
        # TODO: Convert this to use lists? (Will probably make JSON dump weird . . .)
        self.osc_1_mix = 0
        self.osc_2_mix = 0
        self.osc_3_mix = 0
        self.osc_4_mix = 0
        
        if osc1 == 0:
            self.osc_1_mix = osc1mix
        elif osc1 == 1:
            self.osc_2_mix = osc1mix
            self.osc_2_sqpct = sqpct
        elif osc1 == 2:
            self.osc_3_mix = osc1mix
        elif osc1 == 3:
            self.osc_4_mix = osc1mix
        
        if osc2 == 0:
            self.osc_1_mix = osc2mix
        elif osc2 == 1:
            self.osc_2_mix = osc2mix
            self.osc_2_sqpct = sqpct
        elif osc2 == 2:
            self.osc_3_mix = osc2mix
        elif osc2 == 3:
            self.osc_4_mix = osc2mix
        
        self.adsr = [(np.random.random() ** config["adsr_skew_factor"]) * config["adsr_maximum"]["attack"],
                     (np.random.random() ** config["adsr_skew_factor"]) * config["adsr_maximum"]["decay"],
                     config["adsr_minimum"]["sustain"] + (1 - config["adsr_maximum"]["sustain"]) * np.random.random() * config["adsr_maximum"]["sustain"],
                     (config["adsr_minimum"]["release"] + (1 - config["adsr_minimum"]["release"]) * (np.random.random() ** config["adsr_skew_factor"])) * config["adsr_maximum"]["release"]]
        self.lfo_1_freq = np.random.random() * config["lfo_maximum"]["freq"]
        self.lfo_1_depth = (np.random.random() ** config["lfo_skew_factor"]) * config["lfo_maximum"]["depth"]
        self.lfo_2_freq = np.random.random() * config["lfo_maximum"]["freq"]
        self.lfo_2_depth = (np.random.random() ** config["lfo_skew_factor"]) * config["lfo_maximum"]["depth"]

    def render(self, notes: np.array):
        # arr is a double array with notes and pitches
        duration = sum(notes[:, 1]) + self.adsr[0] + self.adsr[1] + self.adsr[3] # Length of the sequence plus the last release
        output = np.zeros(int(duration * sr))
        start_sample = 0
        for i in range(len(notes)):
            freq = m_to_f(notes[i][0])
            dur = notes[i][1] + self.adsr[3]
            if (self.adsr[0] + self.adsr[1] + self.adsr[3] > dur):
                dur = self.adsr[0] + self.adsr[1] + self.adsr[3]

            wave1 = fm_synthesis(freq, sine(self.lfo_1_freq, dur, self.lfo_1_depth), WaveType.SINE)
            wave2 = fm_synthesis(freq, sine(self.lfo_1_freq, dur, self.lfo_1_depth), WaveType.SQUARE, square_pct=self.osc_2_sqpct)
            wave3 = fm_synthesis(freq, sine(self.lfo_1_freq, dur, self.lfo_1_depth), WaveType.SAW)
            wave4 = noise(dur)

            wave1 = apply_envelope(wave1, self.adsr) * self.osc_1_mix
            wave2 = apply_envelope(wave2, self.adsr) * self.osc_2_mix
            wave3 = apply_envelope(wave3, self.adsr) * self.osc_3_mix
            wave4 = apply_envelope(wave4, self.adsr) * self.osc_4_mix
            
            for j in range(len(wave1)):
                output[start_sample + j] += wave1[j] + wave2[j] + wave3[j] + wave4[j]
            start_sample += int(notes[i][1] * sr)

        output = am_synthesis(output, sine(self.lfo_2_freq, duration, self.lfo_2_depth))
        return output
    
    def assign_params_from_array(self, arr: np.array):
        # self.osc_1 = WaveType(np.argmax(arr[0:4]))
        # self.osc_1_sqpct = arr[4] if arr[4] >= 0 and arr[4] <= 1 else 0.5
        # self.osc_2 = WaveType(np.argmax(arr[5:9]))
        # self.osc_2_sqpct = arr[9] if arr[9] >= 0 and arr[9] <= 1 else 0.5
        # self.mix_pct = arr[10] if arr[10] >= 0 and arr[10] <= 1 else 0.5
        # self.adsr = np.abs(arr[11:15])
        # self.lfo_1_freq = abs(arr[15])
        # self.lfo_1_depth = abs(arr[16])
        # self.lfo_2_freq = abs(arr[17])
        # self.lfo_2_depth = abs(arr[18])

        self.osc_1_mix = arr[0] if arr[0] >= 0 and arr[0] <= 1 else 1
        self.osc_2_mix = arr[1] if arr[1] >= 0 and arr[1] <= 1 else 0
        self.osc_3_mix = arr[2] if arr[2] >= 0 and arr[2] <= 1 else 0
        self.osc_4_mix = arr[3] if arr[3] >= 0 and arr[3] <= 1 else 0
        self.osc_2_sqpct = arr[4]
        self.adsr = np.abs(arr[5:9])
        self.lfo_1_freq = abs(arr[10])
        self.lfo_1_depth = abs(arr[11])
        self.lfo_2_freq = abs(arr[12])
        self.lfo_2_depth = abs(arr[13])
        
    
    def as_json(self):
        return json.dumps(self.__dict__)
    