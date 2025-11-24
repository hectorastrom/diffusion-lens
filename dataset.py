from clip_classifier import load_cod10k_lazy
from torchvision import transforms
from torch.utils.data import DataLoader

tensor_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

def pytorch_transform_fn(examples):
    # examples is a dict of lists: {'image': [PIL, ...], 'label': [0, ...]}
    examples['pixel_values'] = [tensor_transform(img.convert("RGB")) for img in examples['image']]
    del examples['image'] # ensures we only have 2 cols for loader
    return examples

def build_COD_torch_dataset(split_name = 'train'):
    dataset = load_cod10k_lazy()[split_name]
    dataset.set_transform(pytorch_transform_fn)
    
    label_feature = dataset.features['label']
    # Adding some metadata that's helpful
    dataset.all_classes = label_feature.names 
    dataset.label2str = label_feature.int2str
    
    return dataset
    
if __name__ == "__main__":
    dataset = build_COD_torch_dataset('train')
    loader = DataLoader(dataset, batch_size=4, num_workers=1, shuffle=True)
    
    for item in loader: # dict{'label', 'pixel_values'}
        img_tensor = item['pixel_values']
        label_int = item['label']
        print("Image tensor:", img_tensor.shape)
        print("Labels:", label_int)
        print("Str labels:", dataset.label2str(label_int))
        break
