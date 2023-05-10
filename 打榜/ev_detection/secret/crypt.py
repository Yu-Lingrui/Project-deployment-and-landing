import os

def model_encrypt(source):
    source = str(source)
    tools_path = os.path.abspath(os.path.dirname(__file__))
    index = source.find(".")
    output = source[:index] + "_encrypt" + source[index:]
    output_dir = os.path.dirname(output)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cmd = '{} {} {}'.format(os.path.join(tools_path, 'ems_encrypt_file'), source, output)
    status = os.system(cmd)
    assert status == 0, 'Something wrong when model encrypt'
    os.remove(source)
    os.rename(output, source)
    # print('Encrypted model path: {}'.format(output))
