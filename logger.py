# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import scipy.misc
import os
import csv
import datetime

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO  # Python 3.x


def embedding_logger(tensor, save_path, meta_data=None):
    embedding_var = tf.Variable(tensor)

    os.makedirs(save_path, exist_ok=True)
    meta_path = None
    if meta_data:
        meta_path = os.path.join(save_path, 'meta.csv')
        with open(meta_path, 'w') as f:
            for w in meta_data:
                f.write(w)
                f.write('\n')

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(save_path, sess.graph)
        sess.run(embedding_var.initializer)
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name
        embedding.metadata_path = meta_path
        projector.visualize_embeddings(writer, config)
        saver_embed = tf.train.Saver([embedding_var])
        saver_embed.save(sess, os.path.join(save_path, 'embedding.ckpt'), 1)

    writer.close()


class Logger(object):

    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            try:
                s = StringIO()
            except:
                s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()


class CsvLogger(object):

    def __init__(self, output_file, headers, append_time=False):
        if not os.path.exists(os.path.dirname(output_file)):
            os.mkdir(os.path.dirname(output_file))
        if append_time:
            time_now = datetime.datetime.now().strftime("d%d-H%H-M%M-S%S")
            self._output_file = output_file + time_now
        else:
            self._output_file = output_file
        self._headers = headers
        self._data = [[] for _ in headers]

    def log(self, *args):
        for ii, arg in enumerate(args):
            self._data[ii].append(arg)
        res = np.array(self._data)
        with open(self._output_file, 'w') as f:
            w = csv.writer(f)
            w.writerow(self._headers)
            w.writerows(res.T)


def read_log(input_file):
    _input_file = input_file
    _data = np.genfromtxt(_input_file, dtype=float, delimiter=',', names=True)
    return _data
