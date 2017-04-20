from random import choice
import names
from datetime import datetime

template = """  - type: {0}
    name: {1}
    # Data related configs:
    vocabulary_size: {2}
    user_count: {3}
    max_title_length: {4}
    validation_data: '{5}'
    training_data: '{6}'
    testing_data: '{7}'
    # Embedding matrix configs:
    embedding_size: {8} # Make sure to match pretrained matrix dimensions
    trainable_matrix: {9}
    use_pretrained: {10}
    pre_trained_matrix: '{11}'
    # Learning configs:
    learning_rate: {12}
    training_epochs: {13}
    batch_size: {14}
    rnn_neurons: {15}
    rnn_unit: '{16}' # Can be 'gru' or 'lstm', default: 'lstm'
    hidden_layers: {17}
    hidden_neurons: {18}
    subreddit_input_neurons: {19}
    use_concat_input: {20}
    pre_train_subreddit: {21}
    # Regularisation configs:
    use_l2_loss: {22}
    l2_factor: {23}
    use_dropout: {24}
    dropout_prob: {25}
    # Validation configs:
    use_constant_limit: {26}
    constant_prediction_limit: {27}\n"""

def rtype():
    return "model-builder"

def name():
    name = "network-"
    x = 0
    while True:
        yield name + names.get_first_name() + names.get_last_name() + "-" + str(x)
        x+=1

def vocabulary_size():
    possibilites = ["10000","12000","15000","17000","19000"]
    return choice(possibilites)

def user_count():
    return choice(["51","6"])

def max_title_length():
    possibilites = ["20","25","30","35","40"]
    return choice(possibilites)

def data_set(nbr_of_users):
    possibilites_5_users = [("validation_data_top_5_subreddit_allvotes.csv"
                            ,"training_data_top_5_subreddit_allvotes.csv"
                            ,"testing_data_top_5_subreddit_allvotes.csv")
                           ]
    possibilites_50_users = [("validation_data_top_50_subreddit_allvotes.csv"
                            ,"training_data_top_50_subreddit_allvotes.csv"
                            ,"testing_data_top_50_subreddit_allvotes.csv"),
                             ("validation_data_top_50_subreddit.csv",
                              "training_data_top_50_subreddit.csv",
                              "testing_data_top_50_subreddit.csv")
                            ]

    val = None
    train = None
    test = None
    if nbr_of_users == "6":
        val,train,test = choice(possibilites_5_users)
    elif nbr_of_users == "51":
        val,train,test = choice(possibilites_50_users)
    return val,train,test

def embedding_size():
    return choice(["100","150","300"])

def trainable_matrix():
    return choice(["true","false"])

def use_pretrained():
    return choice(["true","false"])

def pre_trained_matrix(dim):
    if dim == "100":
        return "vectors.txt"
    elif dim == "150":
        return "vectors150d.txt"
    elif dim == "300":
        return "vectors300d.txt"

def learning_rate():
    possibilites = ["0.05", "0.1", "0.15","0.2","0.3","0.35","0.4","0.5"]
    return choice(possibilites)

def training_epochs():
    return choice(["50"])

def batch_size():
    possibilites = ["25", "50", "100"]
    return choice(possibilites)

def lstm_neurons():
    possibilites = ["150","200","250","300","400"]
    return choice(possibilites)

def rnn_unit():
    return choice(["gru","lstm"])

def subreddit_input_neurons():
    return choice(["50", "100", "200"])

def use_pretrain_subreddit():
    return choice(["true", "false"])

def hidden_layers():
    return choice(["0","1","2","3","4","5"])

def hidden_neurons():
    possibilites = ["150","300","450","600","750"]
    return choice(possibilites)

def use_concat_input():
    return choice(["true","false"])

def use_l2_loss():
    return choice(["true","false"])

def l2_factor():
    possibilites = ["0.01", "0.05", "0.1"]
    return choice(possibilites)

def use_dropout():
    return choice(["true","false"])

def dropout_prob():
    possibilites = ["0.5", "0.75", "0.9"]
    return choice(possibilites)

def use_constant_limit():
    return choice(["true","false"])

def constant_prediction_limit():
    possibilites = ["0.2","0.3","0.4","0.5", "0.6", "0.7", "0.8"]
    return choice(possibilites)


name_generator = name()

def get_random_config():
    embed_size = embedding_size()
    nbr_of_users = user_count()
    val,train,test = data_set(nbr_of_users)
    config = template.format(rtype(),
                         name_generator.__next__(),
                         vocabulary_size(),
                         nbr_of_users,
                         max_title_length(),
                         val,
                         train,
                         test,
                         embed_size,
                         trainable_matrix(),
                         use_pretrained(),
                         pre_trained_matrix(embed_size),
                         learning_rate(),
                         training_epochs(),
                         batch_size(),
                         lstm_neurons(),
                         rnn_unit(),
                         hidden_layers(),
                         hidden_neurons(),
                         subreddit_input_neurons(),
                         use_concat_input(),
                         use_pretrain_subreddit(),
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

