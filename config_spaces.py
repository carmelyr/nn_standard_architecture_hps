#---- Configuration Spaces for Different Models: FCNN, CNN, LSTM, GRU, Transformer ----#

from ConfigSpace import ConfigurationSpace, Integer, Float, Categorical

fcnn_seed = 1
cnn_seed = 1
lstm_seed = 1
gru_seed = 1
transformer_seed = 1

#---- FCNN ----#
def get_fcnn_config_space(seed=fcnn_seed):

    cs = ConfigurationSpace(name="FCNN", seed=seed)

    hidden_units = Integer(name="hidden_units", bounds=(16, 128), log=True)     # hidden units with logarithmic scale

    num_layers = Integer(name="num_layers", bounds=(1, 4))

    dropout_rate = Float(name="dropout_rate", bounds=(0.1, 0.6))

    learning_rate = Float(name="learning_rate", bounds=(1e-4, 1e-1), log=True)  # learning rate with logarithmic scale
    
    activation = Categorical("activation", ["relu", "tanh", "gelu"])

    weight_decay = Float(name="weight_decay", bounds=(1e-5, 1e-2))
    
    cs.add([hidden_units, num_layers, dropout_rate, learning_rate, activation, weight_decay])
    
    return cs

#---- CNN ----#
def get_cnn_config_space(seed=cnn_seed):

    cs = ConfigurationSpace(name="CNN", seed=seed)

    num_filters = Integer(name="num_filters", bounds=(64, 256), log=True)       # number of filters with logarithmic scale
    
    num_layers = Integer(name="num_layers", bounds=(1, 5))

    kernel_size = Integer(name="kernel_size", bounds=(3, 7))

    pooling = Categorical("pooling", ["max", "average"])

    pooling_size = Integer(name="pooling_size", bounds=(1, 2))
    
    dropout_rate = Float(name="dropout_rate", bounds=(0.0, 0.5))

    learning_rate = Float(name="learning_rate", bounds=(1e-4, 1e-2), log=True)      # learning rate with logarithmic scale
    
    activation = Categorical("activation", ["relu", "tanh", "gelu", "elu", "sigmoid"])
    
    cs.add([num_filters, num_layers, kernel_size, pooling, pooling_size, dropout_rate, learning_rate, activation])
    
    return cs

#---- LSTM ----#
def get_lstm_config_space(seed=lstm_seed):

    cs = ConfigurationSpace(name="LSTM", seed=seed)

    hidden_units = Integer(name="hidden_units", bounds=(64, 256), log=True)     # hidden units with logarithmic scale

    num_layers = Integer(name="num_layers", bounds=(1, 3))

    dropout_rate = Float(name="dropout_rate", bounds=(0.0, 0.3))

    learning_rate = Float(name="learning_rate", bounds=(1e-4, 1e-3), log=True)   # learning rate with logarithmic scale

    output_activation = Categorical("output_activation", ["relu", "tanh", "gelu"])

    bidirectional = Categorical("bidirectional", [True, False])     # whether to use bidirectional LSTM

    weight_decay = Float(name="weight_decay", bounds=(0.0, 1e-4))

    cs.add([hidden_units, num_layers, dropout_rate, learning_rate, output_activation, bidirectional, weight_decay])

    return cs

#---- GRU ----#
def get_gru_config_space(seed=gru_seed):

    cs = ConfigurationSpace(name="GRU", seed=seed)

    hidden_units = Integer(name="hidden_units", bounds=(32, 256), log=True)     # hidden units with logarithmic scale

    num_layers = Integer(name="num_layers", bounds=(1, 5))

    dropout_rate = Float(name="dropout_rate", bounds=(0.0, 0.5))

    learning_rate = Float(name="learning_rate", bounds=(1e-5, 1e-2), log=True)      # learning rate with logarithmic scale

    output_activation = Categorical("output_activation", ["relu", "tanh", "gelu", "elu", "sigmoid", "linear"])

    bidirectional = Categorical("bidirectional", [True, False])

    weight_decay = Float(name="weight_decay", bounds=(0.0, 1e-3))

    cs.add([hidden_units, num_layers, dropout_rate, learning_rate, output_activation, bidirectional, weight_decay])

    return cs

#---- Transformer ----#
def get_transformer_config_space(seed=transformer_seed):

    cs = ConfigurationSpace(name="Transformer", seed=seed)

    num_heads = Integer(name="num_heads", bounds=(2, 8))

    hidden_units = Integer(name="hidden_units", bounds=(64, 512), log=True)     # hidden units with logarithmic scale

    ff_dim = Integer(name="ff_dim", bounds=(256, 1024), log=True)               # feedforward dimension with logarithmic scale

    num_layers = Integer(name="num_layers", bounds=(1, 6))

    pooling = Categorical("pooling", ["mean", "max"])

    dropout_rate = Float(name="dropout_rate", bounds=(0.0, 0.3))

    learning_rate = Float(name="learning_rate", bounds=(1e-5, 1e-3), log=True)      # learning rate with logarithmic scale

    activation = Categorical("activation", ["relu", "gelu"])

    weight_decay = Float(name="weight_decay", bounds=(0.0, 1e-4))

    cs.add([num_heads, hidden_units, ff_dim, num_layers, pooling, dropout_rate, learning_rate, activation, weight_decay])

    return cs




