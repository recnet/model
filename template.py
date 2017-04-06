template = """   -type: {0}
    name: {1}
    # Data related configs:
    vocabulary_size: {2}
    user_count: {3}
    max_title_length: {4}
    validation_data: {5}
    training_data: {6}
    testing_data: {7}
    # Embedding matrix configs:
    embedding_size: {8} # Make sure to match pretrained matrix dimensions
    trainable_matrix: {9}
    use_pretrained: {10}
    pre_trained_matrix: {11}
    # Learning configs:
    learning_rate: {12}
    training_epochs: {13}
    batch_size: {14}
    lstm_neurons: {15}
    hidden_layers: {16}
    hidden_neurons: {17}
    use_concat_input: {18}
    # Regularisation configs:
    use_l2_loss: {19}
    l2_factor: {20}
    use_dropout: {21}
    dropout_prob: {22}
    # Validation configs:
    use_constant_limit: {23}
    constant_prediction_limit: {24}"""

def rtype():
    return "model-builder"

def name():
    name = "network"
    x = 0
    while True:
        yield name + str(x)
        x+=1

def vocabulary_size():
    pass

def user_count():
    return input("Enter number of users > ")

def max_title_length():
    pass

def validation_data():
    pass

def training_data():
    pass

def testing_data():
    pass

def embedding_size():
    pass

def trainable_matrix():
    pass

def use_pretrained():
    pass

def pre_trained_matrix():
    pass

def learning_rate():
    pass

def training_epochs():
    pass

def batch_size():
    pass

def lstm_neurons():
    pass

def hidden_layers():
    pass

def hidden_neurons():
    pass

def use_concat_input():
    pass

def use_l2_loss():
    pass

def l2_factor():
    pass

def use_dropout():
    pass

def dropout_prob():
    pass

def use_constant_limit():
    pass

def constant_prediction_limit():
    pass


name_generator = name()
users  = user_count()

def get_random_config():
    config = template.format(rtype(),
                         name_generator.__next__(),
                         vocabulary_size(),
                         users,
                         max_title_length(),
                         validation_data(),
                         training_data(),
                         testing_data(),
                         embedding_size(),
                         trainable_matrix(),
                         use_pretrained(),
                         pre_trained_matrix(),
                         learning_rate(),
                         training_epochs(),
                         batch_size(),
                         lstm_neurons(),
                         hidden_layers(),
                         hidden_neurons(),
                         use_concat_input(),
                         use_l2_loss(),
                         l2_factor(),
                         use_dropout(),
                         dropout_prob(),
                         use_constant_limit(),
                         constant_prediction_limit())
    return config

nbr_of_configs = int(input("Enter number of configs to generate > "))
configs = "\n".join( (get_random_config() for x in range(nbr_of_configs)) )

header = """data:
  path: 'resources/datasets/'

network:\n"""

f = open("config.yaml","w")
f.write(header + configs)
f.close()

