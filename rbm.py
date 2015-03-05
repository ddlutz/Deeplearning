"""
author: Douglas Dlutz

References:
numpy: http://docs.scipy.org/doc/numpy/reference/generated/numpy.random.randn.html#numpy.random.randn
geoffry hintons notes: https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
improved learning of GB-RBM: http://users.ics.aalto.fi/praiko/papers/icann11.pdf

"""
import optparse
import numpy as np
import pprint
import math
import matplotlib.pyplot as plt

def main():
  pp = pprint.PrettyPrinter(indent=4)
  
  optparser = optparse.OptionParser()
  optparser.add_option("-j", "--j", dest="num_hidden", default=1, type="int", help="Number of hidden neurons")
  optparser.add_option("-n", "--n", dest="num_iter", default=10000, type="int", help="Number of training epochs")
  optparser.add_option("--lr", "--lr", dest="lr", default=0.005, type="float", help="learning rate")
  
  (opts, _) = optparser.parse_args()
  
  I = 2
  J = opts.num_hidden
  data = load_data()
  
  W = init_weight_matrix(I, J)
  visible_biases = np.zeros(I)
  hidden_biases = np.zeros(J)
  
  pp.pprint(W)
  plot(data)
  weights, visible_biases, hidden_biases = train(data, W, visible_biases, hidden_biases, I, J, opts.num_iter, opts.lr)
  
  actual_hidden = [reconstruct_hidden(data_row, I, J, weights, hidden_biases) for data_row in data]
  reconstructed_visible = [reconstruct_visible(hv, I, J, weights, visible_biases) for hv in actual_hidden]
  
  show_visible = []
  for i in range(len(reconstructed_visible)):
    new_row = []
    new_row.append(reconstructed_visible[i][0])
    new_row.append(reconstructed_visible[i][1])
    new_row.append(data[i][2])
    show_visible.append(new_row)
  
  plot(show_visible)
  
  

def load_data():
  raw_data = open('nutsbolts.csv').read().split('\n')
  data = []
  for row in raw_data:
    split_row = row.split(',')
    if len(row) > 1:
      data_row = [float(split_row[0]), float(split_row[1]), int(split_row[2])]
      data.append(data_row)
  return data
  
"""
returns: the set of weight vectors after being training
"""
def train(data, weights, visible_biases, hidden_biases, I, J, iterations=100000, lr = 0.005):
  print "Beginning training of RBM\n"
  print "initial weight matrix: "
  print weights
  for iter in range(iterations):
    print "Training iter: " + str(iter) + "\n"
    actual_hidden = [reconstruct_hidden(data_row, I, J, weights, hidden_biases) for data_row in data]
    reconstructed_visible = [reconstruct_visible(hv, I, J, weights, visible_biases) for hv in actual_hidden]
    reconstructed_hidden = [reconstruct_hidden(rv, I, J, weights, hidden_biases) for rv in reconstructed_visible]
    
    for i in range(I):
      
      for j in range(J):
        #Calculate Edata and Emodel for this weight pair
        Edata = 0
        Emodel = 0
        
        for row_index in range(len(data)):
          Edata = Edata + (data[row_index][i] * actual_hidden[row_index][j])
          Emodel = Emodel + (reconstructed_visible[row_index][i] * reconstructed_hidden[row_index][j])
          
        Edata = Edata / float(len(data))
        Emodel = Emodel / float(len(data))
        update = lr * (Edata - Emodel)
        weights[i,j] = weights[i,j] + update
        
    for j in range(J):    
      h_data = 0
      h_recon = 0
      #update Hidden biases
      for row_index in range(len(data)):
        h_data = h_data + actual_hidden[row_index][j]
        h_recon = h_recon + reconstructed_hidden[row_index][j]
      h_data = h_data / float(len(data))
      h_recon = h_recon /  float(len(data))
      hidden_biases[j] = lr * (h_data - h_recon)
    for i in range(I):  
      #update visible biases
      v_data = 0
      v_recon = 0
      #update visible biases
      for row_index in range(len(data)):
        v_data = v_data + data[row_index][i]
        v_recon = v_recon + reconstructed_visible[row_index][i]
      v_data = v_data / float(len(data))
      v_recon = v_recon /  float(len(data))
      visible_biases[i] = lr * (v_data - v_recon)
  
  return weights, visible_biases, hidden_biases
    
  
def logit(input):
  return 1.0 / (1.0 + math.exp(-input))
  
def reconstruct_visible(hidden_vector, num_visible, num_hidden, weights, visible_biases):
  new_visible = []
  for i in range(num_visible):
    mean = 0
    for j in range(num_hidden):
      mean = mean + (hidden_vector[j] * weights[i][j])
    mean = mean + visible_biases[i]
    
    new_visible.append(mean + np.random.randn())
  return np.array(new_visible)

def reconstruct_hidden(visible_vector, num_visible, num_hidden, weights, hidden_biases):
  #generate probabilities for each hidden neuron
  #set to 1 if above a uniform random variable between 0 and 1, o.w. zero
  new_hidden = []
  for j in range(num_hidden):
    mean = 0
    for i in range(num_visible):
      mean = mean + (visible_vector[i] * weights[i][j])
    mean = mean + hidden_biases[j]
    probability = logit(mean)
    value = 0
    if probability > np.random.uniform(0,1):
      value = 1
    new_hidden.append(value)  
  return np.array(new_hidden)
  
"""
Construct weight vector according to Hinton's guide
"""
def init_weight_matrix(num_visible, num_hidden):
  return 0.01 * np.random.randn(num_visible, num_hidden)
  
def plot(data):
  x = [row[0] for row in data]
  y = [row[1] for row in data]
  colors = [row[2] for row in data]
  
  plt.scatter(x, y, s=400, c=colors, alpha=0.5)
  plt.show()

if __name__ == "__main__":
  main()
  