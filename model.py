import tensorflow as tf
import os.path
import glob

checkpoint_path = "checkpoints/model.ckpt"
test = tf.Variable(tf.zeros([200]), name="test")

init_op = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    # Loads models if it exists
    checkpoint_files = glob.glob(checkpoint_path + "*")
    if all([os.path.isfile(file) for file in checkpoint_files]) and checkpoint_files:
        saver.restore(sess, checkpoint_path)
        print("Loaded model from: %s" % checkpoint_path)
    else:
        print("No model found, initializing...")
        sess.run(init_op)

    # Do work ...

    # Save model when done
    save_path = saver.save(sess, checkpoint_path)
    print("Model saved in: %s" % save_path)
