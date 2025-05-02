import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema

def get_tflite_operations(tflite_model_path):
    # Load the TFLite model file
    with open(tflite_model_path, "rb") as f:
        model_data = f.read()

    # Parse the TFLite model
    model = schema.Model.GetRootAsModel(model_data, 0)

    # Get the subgraph (usually only one in most models)
    subgraph = model.Subgraphs(0)
    operators = subgraph.OperatorsLength()

    # Extract all the operation codes used in the model
    op_codes = []
    for i in range(operators):
        op_code_index = subgraph.Operators(i).OpcodeIndex()
        op_code = model.OperatorCodes(op_code_index).BuiltinCode()
        op_codes.append(op_code)

    # Map operation codes to their names using BuiltinOperator.__dict__
    op_names = [key for key, value in schema.BuiltinOperator.__dict__.items() if value in op_codes]
    return set(op_names)  # Return unique operation names

# Path to your TFLite model
tflite_model_path = "wake_vision_model_quantized.tflite"

# Get and print operations
operations = get_tflite_operations(tflite_model_path)
print("Operations used in the TFLite model:")
for op in operations:
    print(op)
