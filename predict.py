from utils import load_checkpoint, load_category_names, process_image, print_results, get_input_args_predict

in_arg = get_input_args_predict()

# Load checkpoint
model = load_checkpoint('checkpoint.pth')

# Image
image_filepath = in_arg.path

# Load category names
cat_to_name = load_category_names(in_arg.category_names)

# Print results
print_results(image_filepath, model, in_arg.gpu, cat_to_name, in_arg.top_k)