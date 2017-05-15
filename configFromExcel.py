import sys
import pandas

template = """  - type: {0}
    name: {1}
    # Data related configs:
    vocabulary_size: {2}
    user_count: {3}
    max_title_length: {6}
    validation_data: '{25}'
    training_data: '{26}'
    testing_data: '{27}'
    # Embedding matrix configs:
    embedding_size: {5} # Make sure to match pretrained matrix dimensions
    trainable_matrix: {21}
    use_pretrained: {23}
    pre_trained_matrix: '{22}'
    # Learning configs:
    learning_rate: {4}
    training_epochs: {14}
    batch_size: {13}
    rnn_neurons: {7}
    rnn_unit: '{8}' # Can be 'gru' or 'lstm', default: 'lstm'
    hidden_layers: {10}
    hidden_neurons: {9}
    subreddit_input_neurons: {11}
    use_concat_input: {12}
    pre_train_subreddit: {24}
    # Regularisation configs:
    use_l2_loss: {15}
    l2_factor: {16}
    use_dropout: {17}
    dropout_prob: {18}
    # Validation configs:
    use_constant_limit: {19}
    constant_prediction_limit: {20}\n"""

header = """data:
  path: 'resources/datasets/'

network:\n"""

def config_string(hyperparamaters):
    hyperparamaters = list(hyperparamaters)
    hyperparamaters[2] = int(hyperparamaters[2])
    hyperparamaters[3] = int(hyperparamaters[3])
    hyperparamaters[6] = int(hyperparamaters[6])
    hyperparamaters[5] = int(hyperparamaters[5])
    hyperparamaters[14] = int(hyperparamaters[14])
    hyperparamaters[13] = int(hyperparamaters[13])
    hyperparamaters[7] = int(hyperparamaters[7])
    hyperparamaters[10] = int(hyperparamaters[10])
    hyperparamaters[9] = int(hyperparamaters[9])
    hyperparamaters[11] = int(hyperparamaters[11])
    return template.format(*hyperparamaters)

def many_configs(data,rows_to_select):
    configs = ( config_string(data.iloc(0)[row]) for row in rows_to_select )
    return "\n".join(configs)


if sys.argv[1] == "--help":
    print("Give name of csv file as first argument and then what rows you want from that csv file")
    print("If you want the first and fourth row from text.csv you write")
    print("configFromExcel text.csv 0 3")
    sys.exit(1)


#give file as command line argumnet
data = pandas.read_csv(sys.argv[1])

# rest are the rows you want as configs, be more flexible later
desired_rows = map(int,sys.argv[2:])

configs = many_configs(data,desired_rows)

filename = "config2.yaml"
f = open(filename,"w")
f.write(header + configs)
f.close()


