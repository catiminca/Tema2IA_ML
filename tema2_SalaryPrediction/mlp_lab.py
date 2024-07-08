from typing import List
import numpy

class Layer:

    def forward(self, x: numpy.ndarray) -> numpy.ndarray:
        
        raise NotImplementedError
        
    def backward(self, x: numpy.ndarray, dy: numpy.ndarray) -> numpy.ndarray:
        raise NotImplementedError
        
    def update(self, *args, **kwargs):
        pass  # If a layer has no parameters, then this function does nothing

class FeedForwardNetwork:
    
    def __init__(self, layers: List[Layer]):
        self.layers = layers
        
    def forward(self, x: numpy.ndarray, train: bool = True) -> numpy.ndarray:
        self._inumpyuts = []
        for layer in self.layers:
            if train:
                self._inumpyuts.append(x)
            x = layer.forward(x)
        return x
    
    def backward(self, dy: numpy.ndarray) -> numpy.ndarray:
        #  Calculati gradientul cu fiecare strat
        # Pasi:
        #   - iterati in ordine inversa prin straturile retelei si apelati pentru fiecare dintre ele metoda backward
        #   - folositi self._inumpyuts salvate la fiecare pas din forward pentru a calcula gradientul cu respectivul strat
        #   - transmiteti mai departe valoarea returnata de metoda backward catre urmatorul strat
        #   - incepeti cu gradientul fata de output (dy, primit ca argument).
        for i, layer in reversed(list(enumerate(self.layers))):
            dy = layer.backward(self._inumpyuts[i], dy)
        return dy
        # del self._inumpyuts
    
    def update(self, *args, **kwargs):
        for layer in self.layers:
            layer.update(*args, **kwargs)

class Linear(Layer):
    
    def __init__(self, insize: int, outsize: int) -> None:
        bound = numpy.sqrt(6. / insize)
        self.weight = numpy.random.uniform(-bound, bound, (insize, outsize))
        self.bias = numpy.zeros((outsize,))
        
        self.dweight = numpy.zeros_like(self.weight)
        self.dbias = numpy.zeros_like(self.bias)
   
    def forward(self, x: numpy.ndarray) -> numpy.ndarray:
        #  calculați ieșirea unui strat liniar
        # x - este o matrice numpy B x M, unde 
        #    B - dimensiunea batchului, 
        #    M - dimensiunea caracteristicilor de intrare (insize)
        # Sugestie: folosiți înmulțirea matricială numpy pentru a implementa propagarea înainte într-o singură trecere
        # pentru toate exemplele din batch
        
        return x @ self.weight + self.bias
    
    def backward(self, x: numpy.ndarray, dy: numpy.ndarray) -> numpy.ndarray:
        # calculați dweight, dbias și returnați dx
        # x - este o matrice numpy B x M, unde 
        #     B - dimensiunea batchului, 
        #     M - dimensiunea caracteristicilor (features) de intrare (insize)
        # dy - este o matrice numpy B x N, unde 
        #     B - dimensiunea batchului, 
        #     N - dimensiunea caracteristicilor (features) de ieșire (outsize)
        # Sugestie: folosiți înmulțirea matricială numpy pentru a implementa propagarea înapoi într-o singură trecere 
        #       pentru self.dweight
        # Sugestie: folosiți numpy.sum pentru a implementa propagarea înapoi într-o singură trecere pentru self.dbias

        self.dweight = x.T @ dy
        self.dbias = numpy.sum(dy, axis=0)
        return dy @ self.weight.T 
    
    def update(self, mode='SGD', lr=0.001, mu=0.9):
        if mode == 'SGD':
            self.weight -= lr * self.dweight
            self.bias -= lr * self.dbias
        else:
            raise ValueError('mode should be SGD, not ' + str(mode))
        
class ReLU(Layer):
    
    def __init__(self) -> None:
        pass
    
    def forward(self, x: numpy.ndarray) -> numpy.ndarray:
        # Calculați ieșirea unei unități liniare rectificate
        return numpy.maximum(x, 0)
    
    def backward(self, x: numpy.ndarray, dy: numpy.ndarray) -> numpy.ndarray:
        # Calculați gradientul față de x
        # x - este o matrice numpy B x M, unde B - dimensiunea batchului, M - dimensiunea caracteristicilor
        # Sugestie: utilizați indexarea logică numpy pentru a determina unde intrarea (x) este negativă
        #       și faceți gradientul 0 pentru acele exemple
        return dy * (x > 0)
    

class CrossEntropy:
    
    def __init__(self):
        pass
    
    def softmax(self, x):
        exps = numpy.exp(x)
        return exps / numpy.sum(exps,axis = 1).reshape(-1,1)

    def forward(self, y: numpy.ndarray, t: numpy.ndarray) -> float:
        #  Calculați probabilitatea logaritmică negativă
        # y - matrice numpy (B, K), unde B - dimensiunea batch-ului, K - numărul de clase (numărul de logaritmi)
        # t - vector numpy (B, ), unde B - dimensiunea batch-ului, care indică clasa corectă
        # Pasi: 
        #   - folositi softmax() pe intrari pentru a transforma logits (y) in probabilitati
        #   - selectati probabilitatile care corespund clasei reale (t)
        #   - calculati -log() peste probabilitati
        #   - impartiti la batch size pentru a calcula valoarea medie peste toate exemplele din batch
        p = self.softmax(y)
        return -numpy.mean(numpy.log(p[numpy.arange(len(p)), t]))
    
    def backward(self, y: numpy.ndarray, t: numpy.ndarray) -> numpy.ndarray:
        # Calculati dl/dy
        # Pasi: 
        #   - calculati softmax(y) pentru a determina probabilitatea ca fiecare element sa apartina clasei i
        #   - ajustati gradientii pentru clasa corecta: aplicati scaderea dL/dy_i = pi - delta_ti conform formulelor de mai sus
        #   - impartiti la batch size pentru a calcula valoarea medie peste toate exemplele din batch

        p = self.softmax(y)
        p[numpy.arange(len(p)), t] -= 1
        return p / len(p)
    

def accuracy_MLP(y: numpy.ndarray, t: numpy.ndarray) -> float:
    # Calculati acuratetea
    # Pasi: 
    # - folosiți numpy.argmax() pentru a afla predictiile retelei
    # - folositi numpy.sum() pentru a numara cate sunt corecte comparand cu ground truth (t)
    # - impartiti la batch size pentru a calcula valoarea medie peste toate exemplele din batch
    return numpy.mean(numpy.argmax(y, axis=1) == t)
