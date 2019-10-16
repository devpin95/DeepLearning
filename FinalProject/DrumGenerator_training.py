# https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5

from music21 import converter, instrument, note, chord
import glob
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Activation, Dense
import os


def train_network():
    notes = get_notes()

    n_vocab = len(set(notes))

    network_input, network_output = prepare_sequence(notes, n_vocab)

    model = create_model(network_input, n_vocab)
    model.fit(network_input, network_output, epochs=1, batch_size=64)


def get_notes():
    notes = []
    DIR = 'groove\\dataset'
    num_files = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    i = 1
    for file in glob.glob('groove\\dataset\\*.mid'):
        midi = converter.parse(file)

        print("(%s/%s) Parsing %s" % (i, num_files, file))

        notes_to_parse = None
        parts = instrument.partitionByInstrument(midi)

        if parts:
            notes_to_parse = parts.parts[0].recurse()
        else:
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
        i = i + 1

        if i > 200:
            break
    return notes


def prepare_sequence(notes, n_vocab):
    sequence_length = 100

    pitchnames = sorted(set(item for item in notes))
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    network_input = network_input / float(n_vocab)

    network_output = np_utils.to_categorical(network_output)

    return network_input, network_output


def create_model(network_input, n_vocab):
    model = Sequential([
        LSTM(256, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True),
        Dropout(0.3),
        LSTM(512, return_sequences=True),
        Dropout(0.3),
        LSTM(256),
        Dense(256),
        Dropout(0.3),
        Dense(n_vocab),
        Activation('softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model


if __name__ == '__main__':
    train_network()
