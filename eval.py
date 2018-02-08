import tensorflow as tf
import train


def eval():
    sess = tf.InteractiveSession()
    
    data_processor = data.DataProcessor(src='train', img_size=train.IMG_SIZE)



def main(_):
    if not FLAGS.checkpoint_file:
        raise ValueError('A checkpoint file must be set for evaluation')

    tf.logging.set_verbosity(tf.logging.INFO)


tf.app.flags.DEFINE_string(
    'checkpoint_file', None,
    'The checkpoint to initialize evaluation from')

FLAGS = tf.app.flags.FLAGS

if __name__ == '__main__':
    tf.app.run()