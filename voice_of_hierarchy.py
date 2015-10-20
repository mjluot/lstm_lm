import numpy
import theano, theano.tensor as T
import random
import time

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
            p[0].parent_choice = 0
            new_layer.append(tn)
        if len(current_layer) > 0:
            new_layer.extend(current_layer)
            current_layer = []
        layers.append(new_layer)
        current_layer = new_layer

    return layers

class Model():

    def __init__(self, tree, size):

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
        wp_val=numpy.asarray(self.rng.uniform(low=-numpy.sqrt(6. / (size + 2)), high=numpy.sqrt(6. / (size + 2)),size=(len(self.nodes), size * 2)),dtype=theano.config.floatX)
        self.wp_matrix = theano.shared(value=wp_val,name='b_soft',borrow=True)
        #Parameter_matrix_b
        self.bp_matrix = theano.shared(value=numpy.zeros((len(self.nodes), 2),dtype=theano.config.floatX),name='b_soft',borrow=True)


    def get_functions(self):

        y = T.lvector()
        n_node_route = self.route_node_matrix[y].T
        n_choice_route = self.route_choice_matrix[y].T
        n_mask = self.mask_matrix[y].T
        x = T.dmatrix()

        def step(r,c,iv):

            bs = self.bp_matrix[r]
            wps = self.wp_matrix[r].reshape((r.shape[0], iv.shape[1], 2))
            #c = T.dot(iv, wps) + bs

            #The funny indexes are because we only want (0,0), (1,1) ... etc out of the dot product below
            #there might be a more enlightened way to go about this
            c_eb = T.dot(iv, wps)[T.arange(bs.shape[0]),T.arange(bs.shape[0])] + bs
            e = T.nnet.softmax(c_eb)
            d = T.nnet.softmax(c_eb)[T.arange(bs.shape[0]), c]

            return bs, wps, e, d

        results, _ = theano.scan(fn=step, sequences = [n_node_route, n_choice_route], non_sequences=[x,], outputs_info = [None, None, None, None])
        #res_f = theano.function([x, y], [results[0], results[1], results[2], results[3]])
        res_f = theano.function([x, y], [results[3]])

        #Yes! Finally! The function works! Now continue it to produce the product.

        log_prob_matrix = T.log(results[3])
        masked_log_probs = log_prob_matrix * n_mask
        log_prob_sums = T.sum(masked_log_probs.T, axis=1)
        cost = -T.mean(log_prob_sums)

        get_mask = theano.function([y], [n_mask])
        get_log_prob = theano.function([x, y], [log_prob_sums])
        get_cost = theano.function([x,y], [cost])
        #res = get_cost(numpy.array([[1,1,1,1,1],[2,2,2,2,2]]) ,[1,4])
        #Works so far, good.

        #Simple sgd
        params = [self.wp_matrix, self.bp_matrix]
        gparams = [T.grad(cost, param) for param in params]
        updates = [(param, param - self.learning_rate * gparam) for param, gparam in zip(params, gparams)]
        train_f = theano.function(inputs=[x, y], outputs=[cost], updates=updates)
        #Seems to work, let's return the function
        #for r in range(20):
        #    print train_f(numpy.array([[1,1,1,1,1],[2,2,2,2,2]]) ,[1,4])
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

    output_size = 5000
    size = 5

    values = range(output_size)#10405)
    tree = build_binary_tree(values)
    print tree
    model = Model(tree, size)
    train_f = model.get_functions()

    #Let's build some random training data
    examples = []

    for v in values:
        for i in range(10):
            examples.append(([random.uniform(-1,1) for x in range(size)], v))

    random.shuffle(examples)

    #Minibatches
    minibatches = []
    for i in range(0, len(examples), 10):
        exs = examples[i:i+10]
        batch = [[],[]]
        for e in exs:
            batch[0].append(e[0])
            batch[1].append(e[1])
        minibatches.append(batch)

    start = time.time()

    for i in range(2):
        costs = []
        for mb in minibatches:
            costs.append(train_f(numpy.array(mb[0]), mb[1]))
        print numpy.mean(costs)

    end = time.time()
    print 'TIME', end - start

    #Baseline
    size = 5
    size_out = 5000
    soft_out = 5000

    x = T.matrix('x')  # data, presented as rasterized images
    y = T.ivector('y')  # labels, presented as 1D vector of [int] labels
    rng = numpy.random.RandomState(1234)
    learning_rate = 0.5
    W = theano.shared(value=numpy.asarray(rng.uniform(low=-numpy.sqrt(6. / (size + 2)), high=numpy.sqrt(6. / (size + 2)),size=(size, soft_out)),dtype=theano.config.floatX), name = 'W_soft', borrow = True)
    b = theano.shared(value=numpy.zeros((soft_out,),dtype=theano.config.floatX),name='b_soft',borrow=True)
    params = [W, b]
    p_y_given_x = T.nnet.softmax(T.dot(x, W) + b)
    cost = -T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]), y])
    g_W = T.grad(cost=cost, wrt=W)
    g_b = T.grad(cost=cost, wrt=b)
    updates = [(W, W - learning_rate * g_W), (b, b - learning_rate * g_b)]
    train_f = theano.function(inputs=[x, y], outputs=[cost], updates=updates)

    #
    start = time.time()
    for i in range(2):
        costs = []
        for mb in minibatches:
            costs.append(train_f(numpy.array(mb[0]), mb[1]))
        print numpy.mean(costs)

    end = time.time()
    print 'TIME', end - start

    #import pdb;pdb.set_trace()
    

