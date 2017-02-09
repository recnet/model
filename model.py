import tensorflow as tf

init_op = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
        sess.run(init_op)
        # Do work ...

        # Save model when done
        save_path = saver.save(sess, "checkpoints/model.ckpt")
        print("Model saved in: %s" % save_path)
