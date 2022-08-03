import argparse
import tf2onnx
import onnxruntime as rt
import tensorflow as tf
from tensorflow.keras.models import load_model

parser = argparse.ArgumentParser(
    description='Export to ONNX model')

parser.add_argument('--i', '--input-model', type=str, default='', 
                    help="Path to input Keras model")
parser.add_argument('--o', '--output-model', type=str, default='', 
                    help="Desired path of converted ONNX model")

args = parser.parse_args()

def main():
    # Load model h5
    model = load_model(args.i)

    # Convert to model onnx
    spec = (tf.TensorSpec((None, 24, 24, 1), tf.float32, name="input"),)
    output_path = args.o

    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)
    output_names = [n.name for n in model_proto.graph.output]

    print('Exporting complete.....................')

if __name__ == "__main__":
	main()