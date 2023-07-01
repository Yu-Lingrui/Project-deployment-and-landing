import onnx
import onnx.helper as helper
import sys
import os

def main():

    # 检查命令行参数是否正确
    if len(sys.argv) < 2:
        print("Usage:\n python v8_transform.py best.onnx")
        return 1

    # 获取模型文件路径
    file = sys.argv[1]
    if not os.path.exists(file):
        print(f"文件不存在: {file}")
        return 1

    # 生成新的文件名
    prefix, suffix = os.path.splitext(file)
    dst = prefix + ".transd" + suffix

    # 加载模型
    model = onnx.load(file)

    # 获取最后一个节点
    node  = model.graph.node[-1]

    # 将最后一个节点的输出名称更改为"pre_transpose"
    old_output = node.output[0]
    node.output[0] = "pre_transpose"

    # 遍历模型的输出，找到与旧输出相同的那个，并创建一个新的输出，将其形状进行转置
    for specout in model.graph.output:
        if specout.name == old_output:
            shape0 = specout.type.tensor_type.shape.dim[0]
            shape1 = specout.type.tensor_type.shape.dim[1]
            shape2 = specout.type.tensor_type.shape.dim[2]
            # print(specout.name)
            new_out = helper.make_tensor_value_info(
                specout.name,
                specout.type.tensor_type.elem_type,
                [0, 0, 0]
            )
            new_out.type.tensor_type.shape.dim[0].CopyFrom(shape0)
            new_out.type.tensor_type.shape.dim[2].CopyFrom(shape1)
            new_out.type.tensor_type.shape.dim[1].CopyFrom(shape2)
            specout.CopyFrom(new_out)

    # 创建一个新的转置节点，并将其添加到模型的节点列表中
    model.graph.node.append(
        helper.make_node("Transpose", ["pre_transpose"], [old_output], perm=[0, 2, 1])
    )
    # 将转置后的模型保存到新文件中，并输出保存的文件路径
    print(f"模型已保存至 {dst}")
    onnx.save(model, dst)

if __name__ == "__main__":
    sys.exit(main())
