import tensorflow as tf
import os.path

checkpoint_path = "checkpoints/model.ckpt"
init_op = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    if os.path.isfile(checkpoint_path):
        saver.restore(sess, checkpoint_path)
    else:
        sess.run(init_op)

    # Do work ...

    # Save model when done
    save_path = saver.save(sess, checkpoint_path)
    print("Model saved in: %s" % save_path)
