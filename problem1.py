import sys
import numpy as np
inputfile=sys.argv[1]
outputfile=sys.argv[2]


class Perceptron:

    def __init__(self, outputter, num_features=2, alpha=0.1):
        self.alpha = alpha
        self.outputter = outputter
        self.weights = np.zeros(num_features + 1)
        self.labelset = None

    def predict_all(self, data: np.array, labels: np.array):
        right, wrong = 0, 0
        for row, expected in zip(data, labels):
            actual = self.predict(row)

            if actual != expected:
                wrong += 1
            else:
                right += 1
        return right, wrong

    def predict(self, data: np.array):
        return self.labelset[1] if np.dot(data, self.weights) > 0 else self.labelset[0]


    def fit_sample(self, training_sample: np.array, desired_value):

        prediction= self.predict(training_sample)
        if prediction > desired_value:
            self.weights -= training_sample
        elif prediction < desired_value:
            self.weights += training_sample

    def run(self, training_set: np.array, d: np.array):
        self.labelset = list(set(d))
        self.labelset.sort()
        for (x, y) in zip(training_set, d):
            self.fit_sample(x, y)

        #open(outputfile, "w") as f:

        self.outputter.process(self, training_set, d, self.labelset)



    def score(self, data, labels):
        r, w = self.predict_all(data, labels)
        print("the correctness is "+ str(r/(r+w)))
        return (r/(r+w))


class Outputter(object):
    def process(self, p: Perceptron, data: np.array, expected_labels: np.array, labels: np.array):
        self.fo.write("%d, %d, %d\n"%(p.weights[1], p.weights[2], p.weights[0]))
        #self.fo.write

    def __init__(self, file_path):
        super().__init__()
        self.fo = open(file_path, "w")
def main():
    input_file=sys.argv[1]
    output_file=sys.argv[2]
    raw_data = np.loadtxt(input_file, delimiter=',')
    p = Perceptron(Outputter(output_file))
    data = raw_data[:, [0, 1]]
    labels = raw_data[:, [2]].flatten()
    row_number = raw_data.shape[0]

    bias_column = np.ones(row_number)
    bias_column.shape = (row_number, 1)
    data = np.hstack((bias_column, data))

    while True:
        p.run(data, labels)
        r,w = p.predict_all(data, labels)
        if w == 0:
            break
    return 0
if __name__ == "__main__":
    main()