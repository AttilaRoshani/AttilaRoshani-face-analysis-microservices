import os
import subprocess
import shutil

def generate_grpc_code():
    # Create output directories if they don't exist
    os.makedirs('protos', exist_ok=True)

    # Generate gRPC code
    subprocess.run([
        'python', '-m', 'grpc_tools.protoc',
        '-I.',
        '--python_out=protos',
        '--grpc_python_out=protos',
        'aggregator.proto'
    ])

    # Create __init__.py in protos directory if it doesn't exist
    init_file = os.path.join('protos', '__init__.py')
    if not os.path.exists(init_file):
        with open(init_file, 'w') as f:
            f.write('from . import aggregator_pb2\n')
            f.write('from . import aggregator_pb2_grpc\n')

if __name__ == '__main__':
    generate_grpc_code() 