def attention_layer(self, inputs, trans_dim, variable_scope):
        with tf.variable_scope(variable_scope, reuse=tf.AUTO_REUSE):
            frames = inputs.shape[1]
            dim = inputs.shape[2]
            input_attn = tf.reshape(inputs, [-1, dim])
            layer1 = tf.layers.dense(input_attn, trans_dim, name='trans_attn',
                                     activation=tf.nn.tanh)
            layer1 = tf.nn.dropout(layer1, self.keep_prob)
            layer2 = tf.layers.dense(layer1, 1, use_bias=False, name='vec_attn')
            input_attn_score = tf.nn.softmax(tf.reshape(layer2, [-1, frames]))
            input_context_vec = tf.reduce_sum(
                tf.expand_dims(input_attn_score, -1) * tf.reshape(inputs, [-1, frames, dim]), 1)
        return input_context_vec, input_attn_score


