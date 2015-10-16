import theano, theano.tensor as T
import numpy
import codecs
from theano import config
import gzip
import sys

minibatch_size = 100
vector_len = 50
max_seq_len = 25

def read_text(filename):

    inf = gzip.open(filename, 'rt').readlines()
    lines = []
    for l in inf[:7000]:
        lines.append(l)
    #inf.close()

    seen_vocab = set()
    vocabulary = []
    sentences = []
    for l in lines:
        sentence = []
        if len(l.split()) < max_seq_len:
            for token in l.split():
                if token not in seen_vocab:
                    seen_vocab.add(token)
                    vocabulary.append(token) 
                sentence.append(vocabulary.index(token))
            sentences.append(sentence)
    return vocabulary, sentences


def make_batches_and_stuff(filename):

    vocabulary, sentences = read_text(filename)
    #Find sentences which are shorter than 25
    #and add padding to make sizes equal
    padded_sents = []
    for s in sentences:
        if len(s) < max_seq_len:
            padding = [len(vocabulary) for i in range(max_seq_len - len(s))]
            padded_sents.append(s + padding)

    #Make into minibatches ready to be trained
    minibatches = []
    #Thanks http://stackoverflow.com/questions/9671224/split-a-python-list-into-other-sublists-i-e-smaller-lists
    chunks=[padded_sents[x:x+minibatch_size] for x in xrange(0, len(padded_sents), minibatch_size)]

    for chunk in chunks:
        inputs = []
        labels = []
        for i in range(0,len(chunk[0]) - 1):
            ip = []
            lbl = []
            for s in chunk:
                ip.append(s[i])
                lbl.append(s[i+1])
            inputs.append(numpy.array(ip, dtype=int))
            labels.append(numpy.array(lbl, dtype=int))

        minibatches.append((inputs, labels))

    return minibatches, vocabulary, sentences


def main():

    if len(sys.argv) > 1 and sys.argv[1] == 'predict':
        if len(sys.argv) > 3:
            predict(sys.argv[2], sys.argv[3])
        else:
            predict('the')
    else:
        train()

def predict(starting_token, model_name='lm_956'):

    #Make vocab, training data and the word vector matrix
    minibatches, vocab, sentences = make_batches_and_stuff('/usr/share/ParseBank/various-datasets/wikipedia-Nov-2013.txt.gz')

    rng = numpy.random.RandomState(1234)
    WV = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (vector_len + len(vocab))),
                    high=numpy.sqrt(6. / (vector_len + len(vocab))),
                    size=(vector_len, len(vocab) + 1)
                ),
                dtype=theano.config.floatX
            )
    W = theano.shared(value=WV, name='W', borrow=True)
    i = T.lvector()
    single_vec = W.T[i]
    get_vector = theano.function([i], single_vec)

    lstm = lstm_block(vector_len, len(vocab) + 1)

    x = T.matrix('x')
    y = T.lvector('y')

    costs = T.vector('costs')

    lstm.W_i, lstm.U_i, lstm.b_i, lstm.W_c, lstm.U_c, lstm.b_c, lstm.W_f, lstm.U_f, lstm.b_f, lstm.W_o, lstm.V_o, lstm.U_o, lstm.b_o, lstm.W_log, lstm.b_log, W = load_model(model_name)

    def step(examples, prev_h_t, prev_mem):

        #Let's get the word_vectors
        i_t = T.nnet.sigmoid(T.dot(W.T[examples], lstm.W_i) + T.dot(prev_h_t, lstm.U_i) + lstm.b_i)
        cc_t = T.tanh(T.dot(W.T[examples], lstm.W_c) + T.dot(prev_h_t, lstm.U_c) + lstm.b_c)

        f_t = T.nnet.sigmoid(T.dot(W.T[examples], lstm.W_f) + T.dot(prev_h_t, lstm.U_f) + lstm.b_f)
        c_t = i_t * cc_t + f_t * prev_mem
        o_t = T.nnet.sigmoid(T.dot(W.T[examples], lstm.W_o) + T.dot(prev_h_t, lstm.U_o) + T.dot(c_t, lstm.V_o) + lstm.b_o)
        h_t = o_t * T.tanh(c_t)

        p_y_given_x = T.nnet.softmax(T.dot(h_t, lstm.W_log) + lstm.b_log)
        y_pred = T.argmax(p_y_given_x, axis=1)

        return y_pred, h_t, c_t

    mem = numpy.zeros((1,vector_len))
    h = numpy.zeros((1,vector_len))
    l = numpy.array(0)
    examples = T.lmatrix()
    labels = T.lmatrix()
    mask = T.dmatrix()
    iput = T.lvector()

    outputs_info = [iput, T.as_tensor_variable(h), T.as_tensor_variable(mem)]
    xresult, _ = theano.scan(fn=step, outputs_info=outputs_info, sequences = [], n_steps=max_seq_len - 1)
    res_f = theano.function([iput], xresult)

    for token in res_f([0])[0]:#vocab.index(starting_token))[0]:
        try:
            print vocab[token[0]]
        except:
            print token






    import pdb; pdb.set_trace()




def train():

    #Make vocab, training data and the word vector matrix
    minibatches, vocab, sentences = make_batches_and_stuff('/usr/share/ParseBank/various-datasets/wikipedia-Nov-2013.txt.gz')

    rng = numpy.random.RandomState(1234)
    WV = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (vector_len + len(vocab))),
                    high=numpy.sqrt(6. / (vector_len + len(vocab))),
                    size=(vector_len, len(vocab) + 1)
                ),
                dtype=theano.config.floatX
            )
    W = theano.shared(value=WV, name='W', borrow=True)
    i = T.lvector()
    single_vec = W.T[i]
    get_vector = theano.function([i], single_vec)

    lstm = lstm_block(vector_len, len(vocab) + 1)

    x = T.matrix('x')
    y = T.lvector('y')

    costs = T.vector('costs')

    def step(examples, labels,l, prev_h_t, prev_mem):

        #Let's get the word_vectors
        i_t = T.nnet.sigmoid(T.dot(W.T[examples], lstm.W_i) + T.dot(prev_h_t, lstm.U_i) + lstm.b_i)
        cc_t = T.tanh(T.dot(W.T[examples], lstm.W_c) + T.dot(prev_h_t, lstm.U_c) + lstm.b_c)

        f_t = T.nnet.sigmoid(T.dot(W.T[examples], lstm.W_f) + T.dot(prev_h_t, lstm.U_f) + lstm.b_f)
        c_t = i_t * cc_t + f_t * prev_mem
        o_t = T.nnet.sigmoid(T.dot(W.T[examples], lstm.W_o) + T.dot(prev_h_t, lstm.U_o) + T.dot(c_t, lstm.V_o) + lstm.b_o)
        h_t = o_t * T.tanh(c_t)

        return l + 1, h_t, c_t

    mem = numpy.zeros((minibatch_size,vector_len))
    h = numpy.zeros((minibatch_size,vector_len))
    l = numpy.array(0)
    examples = T.lmatrix()
    labels = T.lmatrix()
    mask = T.dmatrix()

    outputs_info = [T.as_tensor_variable(l), T.as_tensor_variable(h), T.as_tensor_variable(mem)]
    xresult, _ = theano.scan(fn=step, outputs_info=outputs_info, sequences = [examples, labels], n_steps=max_seq_len - 1)
    push_minibatch = theano.function(inputs=[examples, labels], outputs=xresult)
    predictions = xresult[1].reshape(((max_seq_len -1)*minibatch_size,vector_len))
    #The softmax layer
    p_y_given_x = T.nnet.softmax(T.dot(predictions, lstm.W_log) + lstm.b_log)
    result_function = theano.function(inputs=[examples, labels], outputs=p_y_given_x)  
    y_pred = T.argmax(p_y_given_x, axis=1)

    prediction_function = theano.function([examples, labels], y_pred)

    masked_cost = -T.mean(T.log(p_y_given_x)[T.arange(labels.flatten().shape[0]), labels.flatten()]) * mask.flatten()

    cost_vector = T.log(p_y_given_x)[T.arange(labels.flatten().shape[0]), labels.flatten()]
    masked_cost_vector = T.log(p_y_given_x)[T.arange(labels.flatten().shape[0]), labels.flatten()] * mask.flatten()

    cost = -T.mean(masked_cost_vector)

    learning_rate = 0.9
    params = [lstm.W_i, lstm.U_i, lstm.b_i, lstm.W_c, lstm.U_c, lstm.b_c, lstm.W_f, lstm.U_f, lstm.b_f, lstm.W_o, lstm.V_o, lstm.U_o, lstm.b_o, lstm.W_log, lstm.b_log, W]
    gparams = [T.grad(cost, param) for param in params]
    updates = [(param, param - learning_rate * gparam) for param, gparam in zip(params, gparams)]

    masked_train = theano.function(inputs=[examples, labels, mask], outputs=[cost], updates=updates)

    for epoch in range(9000):

        costs = []
        for minibatch in minibatches:

            if len(minibatch[0][0]) < 100:
                continue

            #Make a mask
            mask = []
            for e in minibatch[0]:
                row = []
                for v in e:
                    if v < len(vocab):
                        row.append(1.0)
                    else:
                        row.append(0.0)
                mask.append(numpy.array(row))

            res2 = masked_train(minibatch[0], minibatch[1], numpy.matrix(mask))
            costs.append(res2)

        print numpy.mean(costs), epoch

        save_model(lstm, W, 'alm_' + str(epoch))

def save_model(lstm, W, name):
    import pickle
    out = open(name, 'wb')
    pickle.dump([lstm.W_i, lstm.U_i, lstm.b_i, lstm.W_c, lstm.U_c, lstm.b_c, lstm.W_f, lstm.U_f, lstm.b_f, lstm.W_o, lstm.V_o, lstm.U_o, lstm.b_o, lstm.W_log, lstm.b_log, W], out)
    out.close()

def load_model(name):
    import pickle
    inf = open(name, 'rb')
    params = pickle.load(inf)
    inf.close()
    return params


#Simple lstm block
class lstm_block():
    def __init__(self, size, soft_out, learning_rate=0.8):

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

        #Having a logistic regression layer is cool
        #Let's copy one
        self.W_log = theano.shared(value=numpy.asarray(self.rng.uniform(low=-numpy.sqrt(6. / (size + 2)), high=numpy.sqrt(6. / (size + 2)),size=(size, soft_out)),dtype=theano.config.floatX), name = 'W_soft', borrow = True)
        self.b_log = theano.shared(value=numpy.zeros((soft_out,),dtype=theano.config.floatX),name='b_soft',borrow=True)
        self.params = [self.W_i, self.U_i, self.b_i, self.W_c, self.U_c, self.b_c, self.W_f, self.U_f, self.b_f, self.W_o, self.V_o, self.U_o, self.b_o, self.W_log, self.b_log]


main()
