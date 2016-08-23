#!/usr/local/bin/python3
import numpy as np
import tensorflow as tf
import json
from functools import wraps
from flask import Flask, request, jsonify


"""
Load a tensorflow model and make it available as a REST service
"""
app = Flask(__name__)

def parse_postget(f):
    @wraps(f)
    def wrapper(*args, **kw):
        try:
            d = dict((key, request.values.getlist(key) if len(request.values.getlist(
                key)) > 1 else request.values.getlist(key)[0]) for key in request.values.keys())
        except BadRequest as e:
            raise Exception("Payload must be a valid json. {}".format(e))
        return f(d)
    return wrapper

def restore(checkpoint_file='linear.chk'):
    saver = tf.train.Saver()
    saver.restore(session, checkpoint_file)
    session.run(tf.initialize_all_variables())
    print("Model restored.")
    return session


@app.route('/model', methods=['GET', 'POST'])
@parse_postget
def apply_model(d):
    tf.reset_default_graph()
    with tf.Session() as session:
        n = 1
        x = tf.placeholder(tf.float32, [n], name='x')
        y = tf.placeholder(tf.float32, [n], name='y')
        m = tf.Variable([1.0], name='m')
        b = tf.Variable([1.0], name='b')
        y = tf.add(tf.mul(m, x), b) # fit y_i = m * x_i + b
        y_act = tf.placeholder(tf.float32, [n], name='y_')

        # minimize sum of squared error between trained and actual.
        error = tf.sqrt((y - y_act) * (y - y_act))
        train_step = tf.train.AdamOptimizer(0.05).minimize(error)

        feed_dict = {x: np.array([float(d['x_in'])]), y_act: np.array([float(d['y_star'])])}
        saver = tf.train.Saver()
        saver.restore(session, 'linear.chk')
        y_i, _, _ = session.run([y, m, b], feed_dict)
    return jsonify(output=float(y_i))

if __name__ == '__main__':
    app.run(debug=True)

