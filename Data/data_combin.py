import json
import random

class Convolutional2DLayer:
    def __init__(self, num_feature_maps, feature_map_size, filter_size, filter_spacing, activation_function):
        self.num_feature_maps = num_feature_maps
        self.feature_map_size = feature_map_size
        self.filter_size = filter_size
        self.filter_spacing = filter_spacing
        self.activation_function = activation_function

    def to_dict(self):
        properties = [
            f"num_feature_maps={self.num_feature_maps}",
            f"feature_map_size={self.feature_map_size}",
            f"filter_size={self.filter_size}",
            f"filter_spacing={self.filter_spacing}"
        ]
        if self.activation_function:  
            properties.append(f"activation_function={self.activation_function}")
        code = f"Convolutional2DLayer({', '.join(properties)})"
        return code
    
    def to_args(self):
        args_list = {
            "num_feature_maps": self.num_feature_maps,
            "feature_map_size": self.feature_map_size,
            "filter_size" : self.filter_size,
            "filter_spacing" : self.filter_spacing
        }
        if self.activation_function:  
            args_list["activation_function"]=self.activation_function
        return args_list
            

class FeedForwardLayer:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes

    def to_dict(self):
        code = (f"FeedForwardLayer(num_nodes={self.num_nodes})")
        return code
    
    def to_args(self):
        args_list = {
            "num_nodes": self.num_nodes
        }
        return args_list
    
class MaxPooling2DLayer:
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def to_dict(self):
        code = (f"MaxPooling2DLayer(kernel_size={self.kernel_size})")
        return code
    
    def to_args(self):
        args_list = {
            "kernel_size" : self.kernel_size
        }
        return args_list
 
class NeuralNetwork:
    def __init__(self, layers, layer_spacing):
        self.layers = layers
        self.layer_spacing = layer_spacing

    def to_dict(self):
        layer_code = []
        layer_args = []
        for layer in self.layers:
            layer_code.append(layer.to_dict())
            layer_args.append(layer.to_args())
        
        code = f"nn=NeuralNetwork({', '.join(layer_code)}, layer_spacing: {self.layer_spacing})"
        return code, layer_args

def generate_nn_config(num_layers):
    layers = []
    for i in range(num_layers):
        layer_type = random.choice([Convolutional2DLayer, FeedForwardLayer, MaxPooling2DLayer])
        if layer_type == Convolutional2DLayer:
            num_feature_maps = random.randint(1, 5)
            feature_map_size = random.randint(3, 7)
            filter_size = random.choice([3, 5, 7])
            filter_spacing = round(random.uniform(0.1, 0.5), 2)
            activation_function = random.choice([None,"ReLU","Sigmoid"])
            layers.append(Convolutional2DLayer(num_feature_maps, feature_map_size, filter_size, filter_spacing, activation_function))
        
        elif layer_type == FeedForwardLayer:
            num_nodes = random.choice([3, 5, 7])
            layers.append(FeedForwardLayer(num_nodes))

        elif layer_type == MaxPooling2DLayer:
            kernel_size = random.randint(1,5)
            layers.append(MaxPooling2DLayer(kernel_size))

    layer_spacing = round(random.uniform(0.1, 0.5), 2)
    sort_order = {'Convolutional2DLayer': 1, 'FeedForwardLayer': 2, 'MaxPooling2DLayer': 3}
    sorted_layers = sorted(layers, key=lambda layer: sort_order[layer.__class__.__name__])
    output = NeuralNetwork(sorted_layers, layer_spacing).to_dict()
    functions_list = [layer.__class__.__name__ for layer in sorted_layers]
    return output, functions_list

dataset = []
def generate_dataset(num_configs):
    for i in range(num_configs):
        num_layers = 5
        nn = generate_nn_config(num_layers)
        nn_config = nn[0][0]
        args_list = nn[0][1]
        functions_list = nn[1]
        dataset.append({"text":"",
                        "code": nn_config,
                        "functions_list": functions_list,
                        "args_list": args_list}) 
    return dataset     

NUM_CONFIGS = 1200  
generate_dataset(NUM_CONFIGS)

with open("nn_dataset.json", "w") as json_file:
        json.dump(dataset, json_file, indent=4)  

print(f"Generated dataset with {NUM_CONFIGS} configurations")
