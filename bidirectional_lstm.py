import theano, theano.tensor as T
import numpy
import codecs
from theano import config
import gzip
import sys
import random


def main():


     #Two embeddings
     #One with linear mapping
     #Bidirectional lstm
    rng = numpy.random.RandomState(1234)
    the_model = lstm_model(input_size=2, embedding_sizes=[(1000,10),(1000,10)], labels=2, bidirectional=True, mapping=[True,False],rng=rng)

    #Testing with a simple minibatch:
    mask_matrix = numpy.zeros((25,10), dtype=numpy.int64)
    for i in range(mask_matrix.shape[1]):
        ones = random.randint(0, mask_matrix.shape[0])
        for b in range(ones):
            mask_matrix[b][i] = 1

    input_matrix1 = numpy.zeros(mask_matrix.shape, dtype=numpy.int64)
    for i in range(input_matrix1.shape[1]):
        for b in range(input_matrix1.shape[0]):
            if mask_matrix[b][i] > 0:
                input_matrix1[b][i] = random.randint(0,999)

    input_matrix2 = numpy.zeros(mask_matrix.shape, dtype=numpy.int64)
    for i in range(input_matrix2.shape[1]):
        for b in range(input_matrix2.shape[0]):
            if mask_matrix[b][i] > 0:
                input_matrix2[b][i] = random.randint(0,999)

    label_matrix = numpy.zeros(mask_matrix.shape, dtype=numpy.int64)
    for i in range(label_matrix.shape[1]):
        for b in range(label_matrix.shape[0]):
            if mask_matrix[b][i] > 0:
                label_matrix[b][i] = random.randint(0,1)

    print 'Inputs'
    print input_matrix1
    print input_matrix2
    print 'Labels'
    print label_matrix
    print 'Mask'
    print mask_matrix
    print
    print 'Prediction without training'
    print the_model.pred_f(input_matrix1, input_matrix2, mask_matrix) * mask_matrix
    print f_score(the_model.pred_f(input_matrix1, input_matrix2, mask_matrix) * mask_matrix, label_matrix * mask_matrix, mask_matrix)
    print

    print 'Training'
    print [the_model.train_f(input_matrix1, input_matrix2, label_matrix, mask_matrix) for i in range(700)][-1]

    print
    print 'Prediction after training'
    print the_model.pred_f(input_matrix1, input_matrix2, mask_matrix) * mask_matrix

    print f_score(the_model.pred_f(input_matrix1, input_matrix2, mask_matrix) * mask_matrix, label_matrix * mask_matrix, mask_matrix)


    import pdb;pdb.set_trace()

def f_score(predicted, labels, mask):

    tp = numpy.sum(predicted & labels)
    fn = 0.0
    fp = 0.0
    for p, l, m in zip(predicted.flatten(), labels.flatten(), mask.flatten()):
        if p < 1 and l > 0 and m > 0:
            fn += 1.0
        if p > 0 and l < 1 and m > 0:
            fp += 1.0
    try:
        return (2*tp)/(2*tp + fp + fn) 
    except:
        return 0.0

class mapping_layer():

    def __init__(self,size,x,rng):
        self.W = theano.shared(value=numpy.asarray(rng.uniform(low=-numpy.sqrt(6. / (size + 2)), high=numpy.sqrt(6. / (size + 2)),size=(size, size)),dtype=theano.config.floatX), name = 'W_soft', borrow = True)
        self.x = x
        self.output = T.dot(self.x, self.W)


class lstm_model():

    def __init__(self, input_size, embedding_sizes, labels, bidirectional, mapping, rng):

        self.size = numpy.sum([e[1] for e in embedding_sizes])
        self.lstm = lstm_block(self.size)

        #If bidirectional, we need another lstm block
        if bidirectional:
            self.lstm2 = lstm_block(self.size)

        #Let's build the embeddings
        self.embeddings = []
        for e in embedding_sizes:
            val = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / numpy.sum(e)),
                    high=numpy.sqrt(6. / numpy.sum(e)),
                    size=e
                ),
                dtype=theano.config.floatX
            )
            self.embeddings.append(theano.shared(val))

        #Inputs
        self.mappings = []
        self.inputs = [T.lmatrix() for i in range(input_size)]
        self.v_outputs = []

        #Either the vectors are fetched directly from the embedding matrix or mapped
        for i, e, m, sizes in zip(self.inputs, self.embeddings, mapping, embedding_sizes):
            if not m:
                self.v_outputs.append(e[i])
            else:
                x = e[i]
                self.mappings.append(mapping_layer(sizes[1],x,rng))
                self.v_outputs.append(self.mappings[-1].output)

        #Concatenation of the vectors
        self.conc = T.concatenate(self.v_outputs, axis=2)

        #Let's make the lstm
        examples = self.conc
        self.label_matrix = T.lmatrix()#imatrix()
        self.mask_matrix = T.lmatrix()#bmatrix()

        #Step function, a single iteration
        def step(examples, maskl, prev_h_t, prev_mem):

            i_t = T.nnet.sigmoid(T.dot(examples, self.lstm.W_i) + T.dot(prev_h_t, self.lstm.U_i) + self.lstm.b_i)
            cc_t = T.tanh(T.dot(examples, self.lstm.W_c) + T.dot(prev_h_t, self.lstm.U_c) + self.lstm.b_c)
            f_t = T.nnet.sigmoid(T.dot(examples, self.lstm.W_f) + T.dot(prev_h_t, self.lstm.U_f) + self.lstm.b_f)
            c_t = i_t * cc_t + f_t * prev_mem
            o_t = T.nnet.sigmoid(T.dot(examples, self.lstm.W_o) + T.dot(prev_h_t, self.lstm.U_o) + T.dot(c_t, self.lstm.V_o) + self.lstm.b_o)
            h_t = o_t * T.tanh(c_t)
            h_t = (h_t.T * maskl).T
            c_t = (c_t.T * maskl).T

            return h_t, c_t, maskl

        examples = self.conc
        outputs_info = [T.zeros_like(self.conc[0]), T.zeros_like(self.conc[0]), None]
        xresult, _ = theano.scan(fn=step, outputs_info=outputs_info, sequences = [examples, self.mask_matrix])

        #push = theano.function([self.inputs[0], self.inputs[1], self.mask_matrix], xresult)

        #Let's build a softmaxlayer
        #Bidirectional
        if bidirectional:

            #Slightly embarrassing, have to circumvent somehow...
            def step2(examples, maskl, prev_h_t, prev_mem):

                i_t = T.nnet.sigmoid(T.dot(examples, self.lstm2.W_i) + T.dot(prev_h_t, self.lstm2.U_i) + self.lstm2.b_i)
                cc_t = T.tanh(T.dot(examples, self.lstm2.W_c) + T.dot(prev_h_t, self.lstm2.U_c) + self.lstm2.b_c)
                f_t = T.nnet.sigmoid(T.dot(examples, self.lstm2.W_f) + T.dot(prev_h_t, self.lstm2.U_f) + self.lstm2.b_f)
                c_t = i_t * cc_t + f_t * prev_mem
                o_t = T.nnet.sigmoid(T.dot(examples, self.lstm2.W_o) + T.dot(prev_h_t, self.lstm2.U_o) + T.dot(c_t, self.lstm2.V_o) + self.lstm2.b_o)
                h_t = o_t * T.tanh(c_t)
                h_t = (h_t.T * maskl).T
                c_t = (c_t.T * maskl).T

                return h_t, c_t, maskl

            flipped_examples = examples[::-1]
            flipped_mask = self.mask_matrix[::-1]
            xresult_flip, _ = theano.scan(fn=step2, outputs_info=outputs_info, sequences = [flipped_examples, flipped_mask])

            #Since in this, bidirectional case we have two outputs from the same sequence we concatenate them
            res_vecs = T.concatenate([xresult[0], xresult_flip[0][::-1]], axis=2)
            self.W_log = theano.shared(value=numpy.asarray(rng.uniform(low=-numpy.sqrt(6. / (self.size + 2)), high=numpy.sqrt(6. / (self.size + 2)),size=(self.size * 2, labels)),dtype=theano.config.floatX), name = 'W_soft', borrow = True)
            self.b_log = theano.shared(value=numpy.zeros((labels,),dtype=theano.config.floatX),name='b_soft',borrow=True)
            self.predictions = (T.dot(res_vecs, self.W_log) + self.b_log)
            #self.rev_push = theano.function([self.inputs[0], self.inputs[1], self.mask_matrix], xresult_flip)

        else:

            self.W_log = theano.shared(value=numpy.asarray(rng.uniform(low=-numpy.sqrt(6. / (self.size + 2)), high=numpy.sqrt(6. / (self.size + 2)),size=(self.size, labels)),dtype=theano.config.floatX), name = 'W_soft', borrow = True)
            self.b_log = theano.shared(value=numpy.zeros((labels,),dtype=theano.config.floatX),name='b_soft',borrow=True)
            self.predictions = (T.dot(xresult[0], self.W_log) + self.b_log) 

        #To make things more simple, let's reshape
        reshaped_predictions = self.predictions.reshape((self.inputs[0].shape[0] * self.inputs[0].shape[1], labels))
        self.probs = T.nnet.softmax(reshaped_predictions)   
        #pf = theano.function([self.inputs[0], self.inputs[1], self.mask_matrix], self.predictions)

        y_pred = T.argmax(reshaped_predictions, axis=1)
        log_probs = T.log(self.probs)[T.arange(self.label_matrix.flatten().shape[0]), self.label_matrix.flatten()] * self.mask_matrix.flatten()

        #Cost
        cost = -1 * (T.sum(log_probs) / T.sum(self.mask_matrix))
        #probf = theano.function([self.inputs[0], self.inputs[1], self.label_matrix, self.mask_matrix], [log_probs, cost])

        #Now for the taining function
        self.learning_rate = theano.shared(numpy.array(0.9))

        params = [self.lstm.W_i, self.lstm.U_i, self.lstm.b_i, self.lstm.W_c, self.lstm.U_c, self.lstm.b_c, self.lstm.W_f, self.lstm.U_f, self.lstm.b_f, self.lstm.W_o, self.lstm.V_o, self.lstm.U_o, self.lstm.b_o]
        params.extend([self.W_log, self.b_log])
        for m in self.mappings:
            params.append(m.W)
        if bidirectional:
            params.extend([self.lstm2.W_i, self.lstm2.U_i, self.lstm2.b_i, self.lstm2.W_c, self.lstm2.U_c, self.lstm2.b_c, self.lstm2.W_f, self.lstm2.U_f, self.lstm2.b_f, self.lstm2.W_o, self.lstm2.V_o, self.lstm2.U_o, self.lstm2.b_o])

        gparams = [T.grad(cost, param) for param in params]
        updates = [(param, param - self.learning_rate * gparam) for param, gparam in zip(params, gparams)]

        inputs_for_train = self.inputs + [self.label_matrix, self.mask_matrix]
        self.train_f = theano.function(inputs=inputs_for_train, outputs=[cost], updates=updates)
        
        #Prediction function
        self.predicted_labels = y_pred.reshape(self.inputs[0].shape)
        inputs_for_pred = self.inputs + [self.mask_matrix]
        self.pred_f = theano.function(inputs_for_pred, self.predicted_labels) 


#Simple lstm block
class lstm_block():
    def __init__(self, size):

        self.size = size
        self.rng = numpy.random.RandomState(1234)

        #W_i -- weight matrix for input
        #U_i -- weight matrix for previous state input(h_i-1)
        #b_i -- input bias vector
        self.W_i = theano.shared(value=numpy.asarray(self.rng.uniform(low=-numpy.sqrt(6. / (size * 2)), high=numpy.sqrt(6. / (size * 2)),size=(size, size)),dtype=theano.config.floatX), name = 'W_i', borrow = True)
        self.U_i = theano.shared(value=numpy.asarray(self.rng.uniform(low=-numpy.sqrt(6. / (size * 2)), high=numpy.sqrt(6. / (size * 2)),size=(size, size)),dtype=theano.config.floatX), name = 'U_i', borrow = True)
        self.b_i = b_values = theano.shared(value=numpy.zeros((size,), dtype=theano.config.floatX), name = 'b_i', borrow=True)
        
        #W_c -- weight matrix for input_1 in memory gate
        #U_c -- weight matrix for input_2 in memory gate
        #b_c -- bias for memory gate
        self.W_c = theano.shared(value=numpy.asarray(self.rng.uniform(low=-numpy.sqrt(6. / (size * 2)), high=numpy.sqrt(6. / (size * 2)),size=(size, size)),dtype=theano.config.floatX), name = 'W_c', borrow = True)
        self.U_c = theano.shared(value=numpy.asarray(self.rng.uniform(low=-numpy.sqrt(6. / (size * 2)), high=numpy.sqrt(6. / (size * 2)),size=(size, size)),dtype=theano.config.floatX), name = 'U_c', borrow = True)
        self.b_c = b_values = theano.shared(value=numpy.zeros((size,), dtype=theano.config.floatX), name = 'b_c', borrow=True)

        #W_f -- weight matrix for input_1 in forget gate
        #U_f -- weight matrix for input_2 in forget gate
        #b_f -- bias for forget gate
        self.W_f = theano.shared(value=numpy.asarray(self.rng.uniform(low=-numpy.sqrt(6. / (size * 2)), high=numpy.sqrt(6. / (size * 2)),size=(size, size)),dtype=theano.config.floatX), name = 'W_f', borrow = True)
        self.U_f = theano.shared(value=numpy.asarray(self.rng.uniform(low=-numpy.sqrt(6. / (size * 2)), high=numpy.sqrt(6. / (size * 2)),size=(size, size)),dtype=theano.config.floatX), name = 'U_f', borrow = True)
        self.b_f = b_values = theano.shared(value=numpy.zeros((size,), dtype=theano.config.floatX), name = 'b_f', borrow=True)

        #W_o -- weight matrix for input_1 in output
        #U_o -- weight matrix for input_2 in output
        #V_o -- weight matrix for memory cells in output
        #b_o -- bias for output
        self.W_o = theano.shared(value=numpy.asarray(self.rng.uniform(low=-numpy.sqrt(6. / (size * 2)), high=numpy.sqrt(6. / (size * 2)),size=(size, size)),dtype=theano.config.floatX), name = 'W_o', borrow = True)
        self.U_o = theano.shared(value=numpy.asarray(self.rng.uniform(low=-numpy.sqrt(6. / (size * 2)), high=numpy.sqrt(6. / (size * 2)),size=(size, size)),dtype=theano.config.floatX), name = 'U_o', borrow = True)
        self.V_o = theano.shared(value=numpy.asarray(self.rng.uniform(low=-numpy.sqrt(6. / (size * 2)), high=numpy.sqrt(6. / (size * 2)),size=(size, size)),dtype=theano.config.floatX), name = 'V_o', borrow = True)
        self.b_o = b_values = theano.shared(value=numpy.zeros((size,), dtype=theano.config.floatX), name = 'b_o', borrow=True)

if __name__ == "__main__":
    main()
