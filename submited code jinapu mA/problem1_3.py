###########  COMS 4701 Jianpu Ma

import sys
import numpy as np

class Perceptron:

    def __init__(self):
        self.weights = np.zeros(3)
        self.it=0

    def predict(self, data):
        return self.labelset[1] if np.dot(data, self.weights) > 0 else self.labelset[0]

    def fitting(self, sample, target):

        prediction= self.predict(sample)
        if prediction > target:
            self.weights -= sample
        elif prediction < target:
            self.weights += sample

    def run(self, training_set, training_label):
        self.labelset = list(set(training_label))
        self.labelset.sort()
        for (x, y) in zip(training_set, training_label):
            self.fitting(x, y)
        self.output(self, training_set, training_label, self.labelset)
        self.it+=1

    def output(self, p, data, expected_labels, labels):
        if self.it==0:
            write_mode="w"
        else:
            write_mode="a"
        self.fo = open(sys.argv[2], write_mode)
        self.fo.write("{},{},{}\n".format(p.weights[1], p.weights[2], p.weights[0]))
        self.fo.close()


    """def score(self, data, labels):
        right, wrong = 0, 0
        for row, expected in zip(data, labels):
            actual = self.predict(row)
            if actual != expected:
                wrong += 1
            else:
                right += 1
        print("Correctness is "+ str(right/(right+wrong)))
        return (right/(right+wrong))"""


def main():
    input_file=sys.argv[1]
    output_file=sys.argv[2]
    raw_data = np.loadtxt(input_file, delimiter=',')

    p = Perceptron()
    data = raw_data[:, [0, 1]]
    labels = raw_data[:, [2]].flatten()
    row = raw_data.shape[0]
    bias_column = np.ones(row)
    bias_column.shape = (row, 1)
    dataset = np.hstack((bias_column, data))
    r, w = 0, 1
    while w!=0:
        w=0
        p.run(dataset, labels)
        for row, expected in zip(dataset, labels):
            actual = p.predict(row)
            if actual != expected:
                w += 1
            else:
                r += 1

if __name__ == "__main__":
    main()