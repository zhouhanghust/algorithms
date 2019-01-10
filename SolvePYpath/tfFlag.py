import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('str_name','str_name_content',"just description")
flags.DEFINE_integer('int_name',999,"desc 2")
flags.DEFINE_boolean('bool_name',False,"desc 3")

FLAGS = tf.app.flags.FLAGS

def show():
    print(FLAGS.str_name)
    print(FLAGS.int_name)
    print(FLAGS.bool_name)

def main(_):
    print("------------------------")
    print(FLAGS.str_name)
    print(FLAGS.int_name)
    print(FLAGS.bool_name)

if __name__ == '__main__':
    show()
    tf.app.run()
