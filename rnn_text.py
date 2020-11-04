import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

"""
Data reader
"""
def read_contents():
    book_data = ''
    # with open('goblet_book.txt', 'r', encoding='utf-8') as book:
    #     lines = book.readlines()
    # for line in lines:
    #     book_data += line
    
    with open('goblet_book.txt', 'r', encoding='utf-8') as book:
        book_contents = book.read()

    book_char = []

    for i in range(len(book_contents)):
        if not(book_contents[i] in book_char):
            book_char.append(book_contents[i])

    print("K = ", len(book_char))

    return book_contents, book_char, len(book_char)

def create_char_to_ind(book_char):
    """
    Maps characters from the alphabet to an index.
    """
    return {k: v for v, k in enumerate(book_char)}

def create_ind_to_char(book_char):
    """
    Maps indexes to characters from the alphabet.
    """
    return {v: k for v, k in enumerate(book_char)}

"""
End of data reader
"""

def softmax(s):
    """
    Activation function that turns our probabilites into values summing to 1.
    Returns all new values for the probabilities.
    """
    e_x = np.exp(s - np.max(s))
    return e_x / e_x.sum(axis=0)

def one_hot_encoding(x, nb_classes):
    """
    Creates one hot vector of length nb_classes.
    """
    return np.transpose(np.eye(nb_classes)[x])

def synthesize_sequence(RNN, n, x, h):
    """
    Synthesizes a sequence of characters.
    """
    Y = np.zeros((RNN.K, n))

    for i in range(n):
        p, h, _ = forward_pass(RNN, x, h)
        label = np.random.choice(RNN.K, p=p[:, 0])

        Y[label][i] = 1
        x = np.zeros(x.shape)
        x[label] = 1

    return Y

def forward_pass(RNN, x, h):
    """
    Computes the forward pass of the RNN.
    """
    ht = h
    tho = x.shape[1]
    H = np.zeros((RNN.m, tho))
    P = np.zeros((RNN.K, tho))
    A = np.zeros((RNN.m, tho))
    for t in range(tho):
        a = np.dot(RNN.W, ht) + np.dot(RNN.U, x[:,[t]]) + RNN.b
        ht = np.tanh(a)
        o = np.dot(RNN.V, ht) + RNN.c
        p = softmax(o)
        H[:, [t]] = ht
        P[:, [t]] = p
        A[:, [t]] = a
    return P, H, A

def compute_loss(P, Y):
    """
    Computes the cross-entropy loss of the RNN.
    """
    loss_sum = 0
    for i in range(P.shape[1]):
        p = P[:, [i]]
        y = Y[:, [i]]
        loss_sum += cross_entropy(p, y)
    return loss_sum

def cross_entropy(Y, P):
    cross_entro = - np.log(np.dot(Y.T,P))
    return cross_entro[0][0]

def backward_pass(RNN, H_params, P, H, A, H0, X, Y):
    """
    Computes the BBTT for the RNN.
    """
    m = H_params['m']
    n = X.shape[1]
    G = -(Y - P)

    grad_c = np.sum(G, axis=1)
    grad_V = np.dot(G, H.T)

    dLdh = np.zeros((n, m))
    dLda = np.zeros((m, n))

    dLdh[-1] = np.dot(G.T[-1], RNN.V)
    dLda[:,-1] = np.multiply(dLdh[-1].T, (1 - np.multiply(np.tanh(A[:, -1]), np.tanh(A[:, -1]))))

    for t in range(n-2, -1, -1):
        dLdh[t] = np.dot(G.T[t], RNN.V) + np.dot(dLda[:, t+1], RNN.W)
        dLda[:,t] = np.multiply(dLdh[t].T, (1 - np.multiply(np.tanh(A[:, t]), np.tanh(A[:, t]))))

    grad_W = np.dot(dLda, H0.T)
    grad_U = np.dot(dLda, X.T)
    grad_b = np.sum(dLda, axis=1)

    dic_grad = {
        'grad_c' : np.reshape(grad_c, (H_params['K'],1)),
        'grad_V' : grad_V, 
        'grad_W' : grad_W,
        'grad_U' : grad_U,
        'grad_b' : np.reshape(grad_b, (m,1))
    }

    #### Clipping gradients to avoid exploiding gradient problem ####
    for name_grad, grad in dic_grad.items():
        grad = np.where(grad<5, grad, 5)
        grad = np.where(grad>-5, grad, -5)
        dic_grad[name_grad] = grad

    return dic_grad

def train(RNN, H_params, book_data):
    # Parameter initialization
    ind_to_char = H_params['ind_to_char']
    char_to_ind = H_params['char_to_ind']
    K = H_params['K']
    m = H_params['m']
    eta = H_params['eta']
    epsilon = H_params['epsilon']
    seq_length = H_params['seq_length']
    smooth_loss = 0
    update_step = 0

    # Gradient initialization
    dic_grad_momentum = {
        'grad_c' : np.zeros((K,1)),
        'grad_V' : np.zeros((K,m)), 
        'grad_W' : np.zeros((m,m)),
        'grad_U' : np.zeros((m,K)),
        'grad_b' : np.zeros((m,1))
    }

    nb_sequences = len(book_data)//seq_length

    # To plot values
    list_loss = []
    list_update_step = []

    for epoch in tqdm(range(H_params['epochs'])):
        e = 0 # Tracking the current position in the book
        hprev = np.zeros((m,1))

        for _ in range(nb_sequences):
            X_chars = book_data[e:e+H_params['seq_length']]
            Y_chars = book_data[e+1:e+H_params['seq_length']+1]
            e += seq_length

            # Reshaping data
            X_index = [char_to_ind[i] for i in X_chars]
            Y_index = [char_to_ind[i] for i in Y_chars]
            X = one_hot_encoding(X_index, K)
            Y = one_hot_encoding(Y_index, K)

            assert X.shape == (K, seq_length)
            assert Y.shape == (K, seq_length)

            P, H1, A = forward_pass(RNN, X, hprev)

            h0 = np.zeros((m,1))
            H0 = np.zeros((m, seq_length))
            H0[:, [0]] = h0
            H0[:, 1:] = H1[:, :-1]
            hprev = H1[:, [-1]]

            dic_grad = backward_pass(RNN, H_params, P, H1, A, H0, X, Y)
            update_step += 1

            for name_grad, grad in dic_grad.items():
                dic_grad_momentum[name_grad] += np.multiply(grad, grad)

            RNN.b -= np.multiply(eta / np.sqrt(dic_grad_momentum['grad_b'] + epsilon), dic_grad['grad_b'])
            RNN.c -= np.multiply(eta / np.sqrt(dic_grad_momentum['grad_c'] + epsilon), dic_grad['grad_c'])
            RNN.U -= np.multiply(eta / np.sqrt(dic_grad_momentum['grad_U'] + epsilon), dic_grad['grad_U'])
            RNN.V -= np.multiply(eta / np.sqrt(dic_grad_momentum['grad_V'] + epsilon), dic_grad['grad_V'])
            RNN.W -= np.multiply(eta / np.sqrt(dic_grad_momentum['grad_W'] + epsilon), dic_grad['grad_W'])

            loss = compute_loss(P, Y)
            if smooth_loss !=0:
                smooth_loss = .999 * smooth_loss + .001 * loss
            else:
                smooth_loss = loss

            if update_step % 999 == 1:
                list_loss.append(smooth_loss)
                list_update_step.append(update_step)

            if update_step % 9999 == 1:
                new_Y = synthesize_sequence(RNN, H_params['new_seq'], X[:,[0]], hprev)
                new_Y_argmax = np.argmax(new_Y, axis = 0)
                new_Y_index = [ind_to_char[i] for i in new_Y_argmax]
                new_Y_str = ''
                for i in range(len(new_Y_index)):
                    new_Y_str+= new_Y_index[i]
                print('-'*10, 'Update Step #', update_step, '-'*10),
                print('Loss : ', smooth_loss)
                print("New Synthesized Sequences : ", new_Y_str)

    plt.plot(list_update_step, list_loss, label = "Smooth loss")
    plt.xlabel('Number of update steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    new_Y = synthesize_sequence(RNN, 1000, X[:,[0]], hprev)
    new_Y_argmax = np.argmax(new_Y, axis = 0)
    new_Y_index = [ind_to_char[i] for i in new_Y_argmax]
    new_Y_str = ''
    for i in range(len(new_Y_index)):
        new_Y_str+= new_Y_index[i]
    print('-'*10, 'Last Update Step #', update_step, '-'*10)
    print('Loss : ', smooth_loss)
    print("*****Final Generated Sequence*****:\n", new_Y_str)

def grad_checking(RNN, H_params, X_chars, Y_chars):
    # Parameter initialization 
    ind_to_char = H_params['ind_to_char']
    char_to_ind = H_params['char_to_ind']
    K = H_params['K']
    m = H_params['m']
    seq_length = H_params['seq_length']

    # Reshaping data
    X_index = [char_to_ind[i] for i in X_chars]
    Y_index = [char_to_ind[i] for i in Y_chars]
    X = one_hot_encoding(X_index, K)
    Y = one_hot_encoding(Y_index, K)

    assert X.shape == (K, seq_length)
    assert Y.shape == (K, seq_length)

    h0 = np.zeros((m,1))

    P, H1, A = forward_pass(RNN, X, h0)

    H0 = np.zeros((m, seq_length))
    H0[:, [0]] = h0
    H0[:, 1:] = H1[:, :-1]

    loss = compute_loss(P, Y)

    grad = backward_pass(RNN, H_params, P, H1, A, H0, X, Y)
    grad2 = computeGradientsNumSlow(H_params, RNN, X, Y, h0)

    rnn_grad_b = grad['grad_b']
    a_grad_b = grad2['grad_b']
    rnn_grad_c = grad['grad_c']
    a_grad_c = grad2['grad_c']
    rnn_grad_U = grad['grad_U']
    a_grad_U = grad2['grad_U']
    rnn_grad_V = grad['grad_V']
    a_grad_V = grad2['grad_V']
    rnn_grad_W = grad['grad_W']
    a_grad_W = grad2['grad_W']
    
    print("\nGradient Differences:")
    
    print("U :", np.sum(np.abs(rnn_grad_U - a_grad_U)))
    print("V :", np.sum(np.abs(rnn_grad_V - a_grad_V)))
    print("W : ", np.sum(np.abs(rnn_grad_W - a_grad_W)))
    print("b :" , np.sum(np.abs(rnn_grad_b - a_grad_b)))
    print("c :", np.sum(np.abs(rnn_grad_c - a_grad_c)))
    
    print("Relative error:\n")
    
    print("U :", np.sum(np.abs(rnn_grad_U - a_grad_U)) / max(1e-6, np.sum(np.abs(rnn_grad_U)) + np.sum(np.abs(a_grad_U))))
    print("V :", np.sum(np.abs(rnn_grad_V - a_grad_V)) / max(1e-6, np.sum(np.abs(rnn_grad_V)) + np.sum(np.abs(a_grad_V))))
    print("W :", np.sum(np.abs(rnn_grad_W - a_grad_W)) / max(1e-6, np.sum(np.abs(rnn_grad_W)) + np.sum(np.abs(a_grad_W))))
    print("b :", np.sum(np.abs(rnn_grad_b - a_grad_b)) / max(1e-6, np.sum(np.abs(rnn_grad_b)) + np.sum(np.abs(a_grad_b))))
    print("c :", np.sum(np.abs(rnn_grad_c - a_grad_c)) / max(1e-6, np.sum(np.abs(rnn_grad_c)) + np.sum(np.abs(a_grad_c))))

def computeGradientsNumSlow(H_params, RNN, X, Y, h0):
    h = 1e-4
    m = H_params['m']
    K = H_params['K']

    grad_b = np.zeros((m, 1))
    grad_c = np.zeros((K, 1))
    grad_U = np.zeros((m, K))
    grad_W = np.zeros((m, m))
    grad_V = np.zeros((K, m))

    print("Computing b gradient")

    for i in range(RNN.b.shape[0]):
        b_true = np.copy(RNN.b)
        b_try = np.copy(RNN.b)
        b_try[i] -= h

        RNN.b = b_try
        P, _, _, = forward_pass(RNN, X, h0)
        c1 = compute_loss(P, Y)

        b_try = np.copy(b_true)
        b_try[i] += h

        RNN.b = b_try
        P, _, _, = forward_pass(RNN, X, h0)
        c2 = compute_loss(P, Y)
        grad_b[i] = (c2 - c1) / (2 * h)
        RNN.b = b_true

    print("Computing c gradient")
    
    for i in range(RNN.c.shape[0]):
        c_true = np.copy(RNN.c)
        c_try = np.copy(RNN.c)
        c_try[i] -= h

        RNN.c = c_try
        P, _, _, = forward_pass(RNN, X, h0)
        c1 = compute_loss(P, Y)

        c_try = np.copy(c_true)
        c_try[i] += h

        RNN.c = c_try
        P, _, _, = forward_pass(RNN, X, h0)
        c2 = compute_loss(P, Y)
        grad_c[i] = (c2 - c1) / (2 * h)
        RNN.c = c_true

    print("Computing V gradient")
    
    for i in range(RNN.V.shape[0]):
        for j in range(RNN.V.shape[1]):
            V_true = np.copy(RNN.V)
            V_try = np.copy(RNN.V)
            V_try[i][j] -= h
            RNN.V = V_try
            P, _, _, = forward_pass(RNN, X, h0)
            c1 = compute_loss(P, Y)

            V_try = np.copy(V_true)
            V_try[i][j] += h
            RNN.V = V_try
            P, _, _, = forward_pass(RNN, X, h0)
            c2 = compute_loss(P, Y)
            grad_V[i][j] = (c2 - c1) / (2 * h)
            RNN.V = V_true

    print("Computing U gradient")
    
    for i in range(RNN.U.shape[0]):
        for j in range(RNN.U.shape[1]):
            U_true = np.copy(RNN.U)
            U_try = np.copy(RNN.U)
            U_try[i][j] -= h
            RNN.U = U_try
            P, _, _, = forward_pass(RNN, X, h0)
            c1 = compute_loss(P, Y)

            U_try = np.copy(U_true)
            U_try[i][j] += h
            RNN.U = U_try
            P, _, _, = forward_pass(RNN, X, h0)
            c2 = compute_loss(P, Y)
            grad_U[i][j] = (c2 - c1) / (2 * h)
            RNN.U = U_true

    print("Computing W gradient")
    
    for i in range(RNN.W.shape[0]):
        for j in range(RNN.W.shape[1]):
            W_true = np.copy(RNN.W)
            W_try = np.copy(RNN.W)
            W_try[i][j] -= h
            RNN.W = W_try
            P, _, _, = forward_pass(RNN, X, h0)
            c1 = compute_loss(P, Y)

            W_try = np.copy(W_true)
            W_try[i][j] += h
            RNN.W = W_try
            P, _, _, = forward_pass(RNN, X, h0)
            c2 = compute_loss(P, Y)
            grad_W[i][j] = (c2 - c1) / (2 * h)
            RNN.W = W_true

    dic_grad = {
    'grad_U' : grad_U,
    'grad_V' : grad_V,
    'grad_W' : grad_W,
    'grad_b' : grad_b,
    'grad_c' : grad_c,
    }

    return dic_grad


class RNN:
    def __init__(self, H_params):
        self.m = H_params['m']
        self.K = H_params['K']

        # Parameters to learn
        self.b = np.zeros((self.m, 1))
        self.c = np.zeros((self.K, 1))
        self.U = np.zeros((self.m, self.K))
        self.W = np.zeros((self.m, self.m))
        self.V = np.zeros((self.K, self.m))

        self.weightInitialisation(0.1)

    def weightInitialisation(self, sig):
        self.U = np.random.normal(0, sig, self.U.shape)
        self.W = np.random.normal(0, sig, self.W.shape)
        self.V = np.random.normal(0, sig, self.V.shape)


if __name__ == "__main__":

    book_contents, book_char, K = read_contents()
    char_to_ind = create_char_to_ind(book_char)
    ind_to_char = create_ind_to_char(book_char)

    H_params = {
        'm' : 100, # Dimensionality of the hidden state
        'eta' : 0.1, # Learning rate
        'seq_length' : 25, # Length of input sequence
        'K' : K,
        'ind_to_char' : ind_to_char,
        'char_to_ind' : char_to_ind,
        'epsilon' : 1e-8,
        'epochs' : 2,
        'new_seq' : 200
    }
    
    # X = book_contents[:params['seq_length']]
    # Y = book_contents[1:params['seq_length']+1]

    rnn_text = RNN(H_params)
    # grad_checking(rnn_text, H_params, X, Y)
    train(rnn_text, H_params, book_contents)