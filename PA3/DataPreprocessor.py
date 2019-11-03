from os import listdir
from os.path import isfile,  join


class DataPreprocessor(object):
    @staticmethod
    def one_hot_character(char, vocab):
        vector = [0] * vocab

        if char > 256:
            if char == '“' or char == '”':
                vector[ord('"')] = 1
            elif char == '’':
                vector[ord('\'')] = 1
        else:
            vector[char] = 1

        return vector

    @staticmethod
    def one_hot_vector(v, vocab):
        x = []
        for char in v:
            x.append(DataPreprocessor.one_hot_character(ord(char), vocab))
        return x

    @staticmethod
    def character_one_hot(v):
        char = ''
        hot = v.index(1)
        return char

    @staticmethod
    def ascii_clean(char):
        c = char
        if char == '“' or char == '”':
            c = '"'
        elif char == '’':
            c = '\''

        return c

    @staticmethod
    def data_targets(dataset, sequence_length):
        sequence_pos = 0
        x, y, seq_x, seq_y = [], [], [], []
        for i in range(0, len(dataset) - 1):
            if sequence_pos == sequence_length:
                sequence_pos = 0
                x.append(seq_x)
                y.append(seq_y)
                seq_x = []
                seq_y = []
            seq_x.append(dataset[i])
            seq_y.append(dataset[i+1])
            sequence_pos = sequence_pos + 1
        return x, y


    @staticmethod
    def get_dataset(dataset, path):
        txts = [f for f in listdir(path) if isfile(join(path, f))]

        for file in txts:
            with open(path + "/" + file, encoding="utf8") as novel:
                print("Reading " + file + "...")
                while True:
                    c = novel.read(1)
                    if not c:
                        break
                    elif c == '':
                        continue
                    dataset.append(DataPreprocessor.ascii_clean(c))
                print("Finished")