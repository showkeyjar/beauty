import tensorflow as tf


def display_model_io(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    print("model ", model_path," io")
    # Print input shape and type
    inputs = interpreter.get_input_details()
    print('\t{} input(s):'.format(len(inputs)))
    for i in range(0, len(inputs)):
        print('\t\t{} {}'.format(inputs[i]['shape'], inputs[i]['dtype']))

    # Print output shape and type
    outputs = interpreter.get_output_details()
    print('\n\t {} output(s):'.format(len(outputs)))
    for i in range(0, len(outputs)):
        print('\t\t{} {}'.format(outputs[i]['shape'], outputs[i]['dtype']))


if __name__=="__main__":
    display_model_io("model_beauty_v1.tflite")
    display_model_io("model_beauty_q_v1.tflite")
