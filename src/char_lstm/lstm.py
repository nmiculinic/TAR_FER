import tensorflow as tf
from model_utils import Model
from sklearn.metrics.pairwise import cosine_similarity


class LSTM(Model):
    max_outputs = 4

    def __init__(self, memory_size, n_layers, out_vector_size, temp, task, model_info):
        super().__init__(*model_info)
        self.memory_size = memory_size
        self.out_vector_size = self.memory_size
        self.n_layers = n_layers
        self.out_vector_size = out_vector_size
        self.temp = temp
        self.task = task

        one_cell = tf.contrib.rnn.BasicLSTMCell
        self.lstm_cell = tf.contrib.rnn.MultiRNNCell(
            [one_cell(self.memory_size, state_is_tuple=False) for _ in range(self.n_layers)])

        self.restore_path = None
        self.saver = None
        self.eval_sess = None
        self.final_state = None
        self.x = None

    def eval_score_tf(self):
        with tf.Session() as sess:
            self.eval_sess = sess
            self.saver.restore(sess, self.restore_path)
            self.logger.info("Restored model {} !!!!".format(self.restore_path))
            graph = sess.graph
            getopt = graph.get_operation_by_name

            self.final_state = getopt("final_state").outputs[0]
            self.x = getopt("x").outputs[0]
            self.eval_score()

    def _predict(self, a, b):
        sem1 = self.eval_sess.run(self.final_state, feed_dict={self.x: a}).flatten().reshape(1, -1)
        sem2 = self.eval_sess.run(self.final_state, feed_dict={self.x: b}).flatten().reshape(1, -1)
        return cosine_similarity(sem1, sem2)[0, 0]

    def preprocess_line(self, sentence):
        return self.task.text_to_indices(sentence)

    def __call__(self, x, initial_state):
        """
        Creates the unrolled RNN specified by cell and transforms it to specified size

        :param x: inputs for all time steps, shape [batch_size, max_time, vector_size]
        :param initial_state:
        :return: outputs for all time steps of shape [batch_size, max_time, out_vector_size]
        """
        with tf.variable_scope("LSTM"):
            # sequence_length = tf.stack([sequence_length for _ in range(self.batch_size)])
            outputs, final_state = tf.nn.dynamic_rnn(self.lstm_cell, x, initial_state=initial_state)
            # outputs is of shape [batch_size, max_time, cell_size] and they're transformed after with a dense layer
            outputs = tf.layers.dense(outputs, self.out_vector_size, name="outputs")

        return outputs, tf.identity(final_state, name="final_state")

    def _define_graph(self, task, optimizer=tf.train.AdamOptimizer()):
        # x shape is [batch_size, seq_len]
        x = tf.placeholder(tf.int32, [None, None], name="x")
        batch_size = tf.shape(x)[0]
        zero_state = self.lstm_cell.zero_state(batch_size, dtype=tf.float32)
        # a better way to initialize this placeholder?
        initial_state = tuple(
            [tf.placeholder_with_default(one_cell_zero_state, [None, 2 * self.memory_size])
             for one_cell_zero_state in zero_state]
        )

        x_one_hot = tf.one_hot(x, depth=task.num_chars, name="x_one_hot")

        outputs, final_state = self(x_one_hot, initial_state)

        one_char = tf.identity(tf.multinomial(outputs[:, -1, :] / self.temp, 1), name="one_char")

        summary_outputs = tf.nn.softmax(outputs, dim=1)

        tf.summary.image("0_Input", tf.expand_dims(x_one_hot, axis=3), max_outputs=LSTM.max_outputs)
        tf.summary.image("0_Network_output", tf.expand_dims(summary_outputs, axis=3),
                         max_outputs=LSTM.max_outputs)

        # We try to predict the 2nd character onward. We don't try to predict the last character from the input
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=x_one_hot[:, 1:, :], logits=outputs[:, :-1, :]), name="loss")
        tf.summary.scalar("Cost", loss)

        optimizer.minimize(loss, name="optimizer")
        tf.identity(tf.summary.merge_all(), name="merged")

        # stupid hack. don't try this at home
        return initial_state

    def train(self, hp, project_path, restore_path=None, optimizer=tf.train.AdamOptimizer()):
        initial_state = self._define_graph(self.task)
        self.saver = tf.train.Saver()

        from numpy import prod, sum
        n_vars = sum([prod(var.shape) for var in tf.trainable_variables()])
        print("This model has", n_vars, "parameters!")

        with tf.Session() as sess:
            graph = sess.graph
            getopt = graph.get_operation_by_name
            getten = graph.get_tensor_by_name

            opt = getopt("optimizer")
            loss = getopt("loss").outputs[0]
            x = getopt("x").outputs[0]
            merged = getopt("merged").outputs[0]
            one_char = getopt("one_char").outputs[0]
            # initial_state = getopt("initial_state").outputs[0]
            final_state = getopt("final_state").outputs[0]

            if restore_path is not None:
                self.saver.restore(sess, restore_path)
                print("Restored model", restore_path, "!!!!!!!!!!!!!")
            else:
                tf.global_variables_initializer().run()
            train_writer = tf.summary.FileWriter(project_path.train_path, sess.graph)
            test_writer = tf.summary.FileWriter(project_path.test_path, sess.graph)

            from time import time
            t = time()

            print("Starting...")
            for step in range(hp.steps + 1):
                data_batch = self.task.next_batch(batch_size=hp.batch_size, seq_len=hp.seq_len)

                _, cost_value = sess.run([opt, loss], feed_dict={x: data_batch})

                if step % 100 == 0:
                    summary = sess.run(merged, feed_dict={x: data_batch})
                    train_writer.add_summary(summary, step)

                    test_data_batch = self.task.next_batch(batch_size=hp.batch_size, seq_len=hp.seq_len, test=True)
                    test_summary = sess.run(merged, feed_dict={x: test_data_batch})
                    test_writer.add_summary(test_summary, step)

                    print("\n ------------------------------- \n")
                    print("Summary generated. Step", step,
                          " Test cost == %.9f Time == %.2fs" % (cost_value, time() - t))
                    print("\n ------------------------------- \n")
                    t = time()

                    sentence, _ = self.task.sample_text(sess, one_char, final_state, x, initial_state,
                                                        length=4 * hp.seq_len)
                    print(sentence)

                    if step % 1000 == 0:
                        self.saver.save(sess, project_path.model_path, global_step=step)
                        print("Model saved!")
