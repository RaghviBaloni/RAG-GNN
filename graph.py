import torch

# Specify the path to the .pt file
file_path = 'dataset/goodreads/graphs/100059.pt'

# Load the object from the .pt file
try:
    loaded_object = torch.load(file_path)
    # Print the type of the loaded object
    print("Type of loaded object:", type(loaded_object))
    
    # If the loaded object is a tensor, print its shape and contents
    if isinstance(loaded_object, torch.Tensor):
        print("Tensor Shape:", loaded_object.shape)
        print("Tensor Contents:", loaded_object)
except Exception as e:
    print("Error loading or inspecting the object:", e)