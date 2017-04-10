from random import choice

template = """   -type: {0}
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
    possibilites = ["10000","12000","15000","17000","19000"]
    return choice(possibilites)

def user_count():
    return choice(["50","5"])

def max_title_length():
    possibilites = ["20","25","30","35","40"]
    return choice(possibilites)

def data_set(nbr_of_users):
    possibilites_5_users = [("validation_data_top_5_subreddit_allvotes.csv"
                            ,"training_data_top_5_subreddit_allvotes.csv"
                            ,"testing_data_top_5_subreddit_allvotes.csv"),
                            ("validation_data_top_5_subreddit_allvotes_binarycontent.csv",
                             "training_data_top_5_subreddit_allvotes_binarycontent.csv",
                             "testing_data_top_5_subreddit_allvotes_binarycontent.csv"),
                            ("validation_data_top_n.csv",
                             "training_data_top_n.csv",
                             "testing_data_top_n.csv")
                           ]
    possibilites_50_users = [("validation_data_top_50_subreddit_allvotes.csv"
                            ,"training_data_top_50_subreddit_allvotes.csv"
                            ,"testing_data_top_50_subreddit_allvotes.csv"),
                            ("validation_data_top_50_subreddit_allvotes_binarycontent.csv",
                             "training_data_top_50_subreddit_allvotes_binarycontent.csv",
                             "testing_data_top_50_subreddit_allvotes_binarycontent.csv"),
                             ("validation_data_top_50_subreddit.csv",
                              "training_data_top_50_subreddit.csv",
                              "testing_data_top_50_subreddit.csv")
                            ]

    val = None
    train = None
    test = None
    if nbr_of_users == "5":
        val,train,test = choice(possibilites_5_users)
    elif nbr_of_users == "50":
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
    possibilites = ["0.15","0.2","0.3","0.35","0.4","0.5"]
    return choice(possibilites)

def training_epochs():
    return choice(["50"])

def batch_size():
    possibilites = ["20","25","30","35","40"]
    return choice(possibilites)

def lstm_neurons():
    possibilites = ["100","150","175","200","250","300"]
    return choice(possibilites)

def hidden_layers():
    return choice(["1","2","3","4","5"])

def hidden_neurons():
    possibilites = ["150","200","250","300","350"]
    return choice(possibilites)

def use_concat_input():
    return choice(["true","false"])

def use_l2_loss():
    return choice(["true","false"])

def l2_factor():
    possibilites = ["0.01"]
    return choice(possibilites)

def use_dropout():
    return choice(["true","false"])

def dropout_prob():
    possibilites = ["0.65","0.70","0.75","0.80"]
    return choice(possibilites)

def use_constant_limit():
    return choice(["true","false"])

def constant_prediction_limit():
    possibilites = ["0.25","0.3","0.35","0.40"]
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
                         embedding_size(),
                         trainable_matrix(),
                         use_pretrained(),
                         pre_trained_matrix(embed_size),
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

