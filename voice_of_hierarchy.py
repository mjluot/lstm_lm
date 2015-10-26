import numpy
import theano, theano.tensor as T
import random
import time
import theano.sparse
import theano.sandbox

#Tree stuff

class TreeNode():

    def __init__(self, index=None, right=None, left=None, parent=None, parent_choice=None):

        self.index = index
        self.right = right
        self.left = left
        self.parent = parent
        self.parent_choice = parent_choice

    def __repr__(self):
        return '<' + str(self.index) + ', 0:' + str(self.left.index) + ', 1:' + str(self.right.index) + '>'

class ResultNode():

    def __init__(self, value=None, parent=None):

        self.value = value
        self.parent = parent
        self.index = 'res:' + str(self.value)

    def __repr__(self):
        return '<' + str(self.value) + '>'

def build_binary_tree(values):

    current_layer = []
    for v in values:
        current_layer.append(ResultNode(value=v))
    layers = [current_layer,]
    count = 0
    while(len(current_layer) > 1):
        pairs = []
	if len(current_layer) > 1:
		while(len(current_layer) > 1):
		    pairs.append(current_layer[:2])
		    current_layer = current_layer[2:]
	else:
		pairs = [current_layer]
                current_layer = []
        new_layer = []
        for p in pairs:
            tn = TreeNode(index=count, right=p[1], left=p[0])
            count += 1
            p[1].parent = tn
            p[1].parent_choice = 1
            p[0].parent = tn
            p[0].parent_choice = -1
            new_layer.append(tn)
        if len(current_layer) > 0:
            new_layer.extend(current_layer)
            current_layer = []
        layers.append(new_layer)
        current_layer = new_layer

    return layers

class Model():

    def __init__(self, tree, size, mb_size):

        self.mb_size = mb_size
        self.learning_rate = 0.5
        self.size = size
        self.rng = numpy.random.RandomState(1234)
        #Make routes
        self.tree = tree

        self.nodes = []
        self.node_dict = {}
        self.result_dict = {}

        self.routes = []

        for layer in tree:
            for i in layer:
                if isinstance(i, TreeNode):
                    self.nodes.append(i)
                    self.node_dict[i.index] = i

                if isinstance(i, ResultNode):
                    self.result_dict[i.value] = i

        self.max_route_len = 0
        for u in sorted(self.result_dict.keys()):
            self.routes.append(self.get_route(self.result_dict[u]))
            if len(self.routes[-1]) > self.max_route_len:
                self.max_route_len = len(self.routes[-1])

        self.route_node_matrix_val = numpy.zeros((len(self.result_dict.keys()), self.max_route_len), dtype=numpy.int)
        self.route_choice_matrix_val = numpy.zeros((len(self.result_dict.keys()), self.max_route_len, ), dtype=numpy.int)
        self.mask_matrix_val = numpy.zeros((len(self.result_dict.keys()), self.max_route_len, ), dtype=numpy.int)

        #import pdb; pdb.set_trace()
        #Routes matrix
        #Mask-matrix
        for i, route in enumerate(self.routes):
            for a in range(self.max_route_len):
                try:
                    self.route_node_matrix_val[i][a] = route[a][0].index
                    self.route_choice_matrix_val[i][a] = route[a][1]
                    self.mask_matrix_val[i][a] = 1.0
                except:
                    self.route_node_matrix_val[i][a] = 0
                    self.route_choice_matrix_val[i][a] = 0
                    self.mask_matrix_val[i][a] = 0.0

        self.route_node_matrix = theano.shared(value=self.route_node_matrix_val, name = 'route_node_matrix', borrow = True)
        self.route_choice_matrix = theano.shared(value=self.route_choice_matrix_val, name = 'route_choice_matrix', borrow = True)
        self.mask_matrix = theano.shared(value=self.mask_matrix_val, name = 'route_mask_matrix', borrow = True)

        #Parameter_matrix_W
        #Make this a little nicer
        wp_val=numpy.asarray(self.rng.uniform(low=-numpy.sqrt(6. / (size + 2)), high=numpy.sqrt(6. / (size + 2)),size=(len(self.nodes), size)),dtype=theano.config.floatX)
        self.wp_matrix = theano.shared(value=wp_val,name='V_soft',borrow=True)
        #Parameter_matrix_b
        #self.bp_matrix = theano.shared(value=numpy.zeros((len(self.nodes), 2),dtype=theano.config.floatX),name='b_soft',borrow=True)


    def get_functions(self, xs=None, ys=None):

        y = T.lvector()
        n_node_route = self.route_node_matrix[y]
        n_choice_route = self.route_choice_matrix[y]
        n_mask = self.mask_matrix[y]
        x = T.dmatrix()

        #1.
        nodes = self.route_node_matrix[y]
        choices = self.route_choice_matrix[y]
        mask = self.mask_matrix[y]

        #2.
        wp = self.wp_matrix[nodes]

        #3. Let's make the gemv

        batch_size = x.shape[0]
        vec_size = x.shape[1]
        route_size = n_choice_route.shape[1]

        #output shape
        o = T.zeros((batch_size, route_size, 1))
        #let's 
        ewps = wp.reshape((batch_size, route_size, vec_size, 1))
        ewp = theano.function([x, y], ewps)

        #Check these
        idx = T.arange(batch_size).reshape((batch_size,1))
        ebin = []
        for i in range(self.mb_size):
            ebin.append(numpy.arange(self.max_route_len))
        odx = T.as_tensor_variable(numpy.asarray(ebin)) 

        iv = x.reshape((x.shape[0], 1, x.shape[1]))

        gb = theano.sandbox.blocksparse.SparseBlockGemv()
        node = gb.make_node(o, ewps, iv, idx, odx)

        matrix_f = theano.function([x, y], node.outputs[0])

        #The dots are done, now is time of direction and the mask
        dots_with_choice = node.outputs[0].reshape((batch_size ,route_size)) * choices
        log_sig = T.log(T.nnet.sigmoid(dots_with_choice)) * mask

        sums = T.sum(log_sig, axis=1)
        cost = -T.mean(sums)
        params = [self.wp_matrix,]

        gparams = [T.grad(cost, param) for param in params]
        updates = [(param, param - self.learning_rate * gparam) for param, gparam in zip(params, gparams)]
        train_f = theano.function(inputs=[x, y], outputs=[cost], updates=updates)

        #import pdb;pdb.set_trace()

        return train_f

    def get_route(self, i):
        route = []
        parent = i.parent
        parent_choice = i.parent_choice
        route.append((parent, parent_choice))
        while(parent != None):
            n_parent = parent.parent
            if n_parent != None:
                parent_choice = parent.parent_choice
                route.append((n_parent, parent_choice))
            parent = parent.parent #Hahaha :D
        return route

if __name__ == "__main__":

    output_size = 50000
    size = 100
    mb_size = 10
    values = range(output_size)#10405)
    tree = build_binary_tree(values)
    #print tree
    model = Model(tree, size, mb_size)
    #Let's build some random training data
    examples = []

    for v in values:
        for i in range(2):
            examples.append(([random.uniform(-1,1) for x in range(size)], v))

    random.shuffle(examples)

    #Minibatches
    minibatches = []
    for i in range(0, len(examples), mb_size):
        exs = examples[i:i+mb_size]
        batch = [[],[]]
        for e in exs:
            batch[0].append(e[0])
            batch[1].append(e[1])
        minibatches.append(batch)

    #mb = minibatches[0]
    #train_f = model.get_functions(numpy.array(mb[0]), mb[1])
    train_f = model.get_functions()
    #Baseline
    x = T.matrix('x')  # data, presented as rasterized images
    y = T.ivector('y')  # labels, presented as 1D vector of [int] labels
    rng = numpy.random.RandomState(1234)
    learning_rate = 0.5
    W = theano.shared(value=numpy.asarray(rng.uniform(low=-numpy.sqrt(6. / (size + 2)), high=numpy.sqrt(6. / (size + 2)),size=(size, soft_out)),dtype=theano.config.floatX), name = 'W_soft', borrow = True)
    b = theano.shared(value=numpy.zeros((output_size,),dtype=theano.config.floatX),name='b_soft',borrow=True)
    params = [W, b]
    p_y_given_x = T.nnet.softmax(T.dot(x, W) + b)
    cost = -T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]), y])
    g_W = T.grad(cost=cost, wrt=W)
    g_b = T.grad(cost=cost, wrt=b)
    updates = [(W, W - learning_rate * g_W), (b, b - learning_rate * g_b)]
    train_nf = theano.function(inputs=[x, y], outputs=[cost], updates=updates)

    #
    start = time.time()
    for i in range(2):
        costs = []
        f_times = []
        nf_times = []

        for mb in minibatches[:2]:
            start = time.time()
            cost = train_f(numpy.array(mb[0]), mb[1])
            end = time.time()
            f_times.append(end - start)

            start = time.time()
            train_nf(numpy.array(mb[0]), mb[1])
            end = time.time()
            nf_times.append(end - start)


    print numpy.mean(f_times), numpy.mean(nf_times)
    

