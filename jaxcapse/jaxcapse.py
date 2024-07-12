from typing import Sequence, List

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import json
import importlib.util
jax.config.update("jax_enable_x64", True)

class MLP(nn.Module):
  features: Sequence[int]
  activations: List[str]
  in_MinMax: np.array
  out_MinMax: np.array
  NN_params: dict
  postprocessing: callable
  emulator_description: dict

  @nn.compact
  def __call__(self, x):
    for i, feat in enumerate(self.features[:-1]):
      if self.activations[i] == "tanh":
        x = nn.tanh(nn.Dense(feat)(x))
      elif self.activations[i] == "relu":
        x = nn.relu(nn.Dense(feat)(x))
      # Add more activation functions as needed
    x = nn.Dense(self.features[-1])(x)
    return x

  def maximin_input(self, input):
    return (input - self.in_MinMax[:,0]) / (self.in_MinMax[:,1] - self.in_MinMax[:,0])

  def inv_maximin_output(self, output):
    return output * (self.out_MinMax[:,1] - self.out_MinMax[:,0]) + self.out_MinMax[:,0]

  def get_Cl(self, input):
    norm_input = self.maximin_input(input)
    norm_model_output = self.apply(self.NN_params, norm_input)
    model_output = self.inv_maximin_output(norm_model_output)
    processed_model_output = self.postprocessing(input, model_output)
    #here we are also postprocessing the Cls, according to what was done in Capse.jl release paper
    return processed_model_output

def get_flax_params(nn_dict, weights):
    in_array, out_array = get_in_out_arrays(nn_dict)
    i_array = get_i_array(in_array, out_array)
    params = [get_weight_bias(i_array[j], in_array[j], out_array[j], weights, nn_dict) for j in range(nn_dict["n_hidden_layers"]+1)]
    layer = ["layer_" + str(j) for j in range(nn_dict["n_hidden_layers"]+1)]
    return dict(zip(layer, params))

def get_weight_bias(i, n_in, n_out, weight_bias, nn_dict):
    weight = np.reshape(weight_bias[i:i+n_out*n_in], (n_in, n_out))
    bias = weight_bias[i+n_out*n_in:i+n_out*n_in+n_out]
    i += n_out*n_in+n_out
    return {'kernel': weight, 'bias': bias}, i

def get_in_out_arrays(nn_dict):
    n = nn_dict["n_hidden_layers"]
    in_array = np.zeros(n+1, dtype=int)
    out_array = np.zeros(n+1, dtype=int)
    in_array[0] = nn_dict["n_input_features"]
    out_array[-1] = nn_dict["n_output_features"]
    for i in range(n):
        in_array[i+1] = nn_dict["layers"]["layer_" + str(i+1)]["n_neurons"]
        out_array[i] = nn_dict["layers"]["layer_" + str(i+1)]["n_neurons"]
    return in_array, out_array

def get_i_array(in_array, out_array):
    i_array = np.empty_like(in_array)
    i_array[0] = 0
    for i in range(1, len(i_array)):
        i_array[i] = i_array[i-1] + in_array[i-1]*out_array[i-1] + out_array[i-1]
    return i_array

def load_weights(nn_dict, weights):
    in_array, out_array = get_in_out_arrays(nn_dict)
    i_array = get_i_array(in_array, out_array)
    variables = {'params': {}}
    i = 0
    for j in range(nn_dict["n_hidden_layers"]+1):
        layer_params, i = get_weight_bias(i_array[j], in_array[j], out_array[j], weights, nn_dict)
        variables['params']["Dense_" + str(j)] = layer_params
    return variables
    
def load_activation_function(nn_dict):
    list_activ_func = []
    for j in range(nn_dict["n_hidden_layers"]):
        list_activ_func.append(nn_dict["layers"]["layer_" + str(j+1)]["activation_function"])
    return list_activ_func

def load_number_neurons(nn_dict):
    list_n_neurons = []
    for j in range(nn_dict["n_hidden_layers"]):
        list_n_neurons.append(nn_dict["layers"]["layer_" + str(j+1)]["n_neurons"])
    list_n_neurons.append(nn_dict["n_output_features"])
    return list_n_neurons

def load_preprocessing(root_path, filename):
    spec = importlib.util.spec_from_file_location(filename, root_path + "/" + filename + ".py")
    test = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test)
    return test.postprocessing

def load_emulator(folder_path):
    in_MinMax = jnp.load(folder_path + "inMinMax.npy")

    f = open(folder_path + '/configuration.json')
    
    # returns JSON object as
    # a dictionary
    NN_dict = json.load(f)
    f.close()
    
    #spec = importlib.util.spec_from_file_location("postprocessing", "postprocessing.py")
    #test = importlib.util.module_from_spec(spec)
    #spec.loader.exec_module(test)
    
    postprocessing = load_preprocessing(folder_path, "postprocessing")
    
    activation_function_list = load_activation_function(NN_dict)
    list_n_neurons = load_number_neurons(NN_dict)
    
    capse_weights = jnp.load(folder_path + "weights.npy")
    out_MinMax = jnp.load(folder_path + "outMinMax.npy")
    variables = load_weights(NN_dict, capse_weights)
    return MLP(list_n_neurons, activation_function_list, in_MinMax, out_MinMax, variables, postprocessing, NN_dict["emulator_description"])
