import argparse
import sys
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from model import efficientdet
from losses import focal

def parse_args(args):
    """
    Parse the arguments.
    """

    parser = argparse.ArgumentParser(description='Exporting the model to a pb file.')
    parser.add_argument('--weights_path', help='The path to h5 file containing the weights.')
    parser.add_argument('--phi', help='Hyper parameter phi', default=0, type=int, choices=(0, 1, 2, 3, 4, 5, 6))
    print(vars(parser.parse_args(args)))
    return parser.parse_args(args)

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph

def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    if (not args.weights_path):
        raise ValueError('Please enter the path to the weights.')

    model, prediction_model = efficientdet(args.phi, num_classes=90, weighted_bifpn=False,
                                            score_threshold=0.01)
    prediction_model.load_weights(args.weights_path, by_name=True)
    frozen_graph = freeze_session(K.get_session(),
                                output_names=[out.op.name for out in prediction_model.outputs])

    tf.train.write_graph(frozen_graph, "prediction_model", "efficientdet-d{}.pb".format(str(args.phi)), as_text=False)

    print("Model was exported succesfully. You can find the resulting pb file in the prediction_model folder.")

if __name__ == '__main__':
    main()
