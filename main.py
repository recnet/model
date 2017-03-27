import tensorflow as tf
import model.basic.model as model
from model.util.networkconfig import yamlconfig as networkconfig

def main():
    """ A main method that creates the model and starts training it """
    with tf.Session() as sess:
        config = 2
        m = model.Model(networkconfig[config], sess)
        m.train()
        m.close_writers()

if __name__ == "__main__":
    main()
