import io
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import numpy as np
import streamlit as st
import PIL

labels = ['level_0', 'level_1', 'level_2', 'normal']

# Recreate the exact fastai v1 cnn_learner(resnet18) head architecture
class AdaptiveConcatPool2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool2d(1)
        self.mp = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


def create_fastai_resnet18(num_classes=4):
    backbone = models.resnet18(weights=None)
    body = nn.Sequential(*list(backbone.children())[:-2])
    head = nn.Sequential(
        AdaptiveConcatPool2d(),
        nn.Flatten(),
        nn.BatchNorm1d(1024),
        nn.Dropout(p=0.25),
        nn.Linear(1024, 512),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.5),
        nn.Linear(512, num_classes)
    )
    return nn.Sequential(body, head)


@st.cache_resource
def load_model():
    model = create_fastai_resnet18(num_classes=4)
    state = torch.load('./data/models/best_resnet.pth', map_location='cpu')
    # fastai v1 saves {'model': state_dict, 'opt': ...}
    weights = state['model'] if 'model' in state else state
    model.load_state_dict(weights)
    model.eval()
    return model


transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


def load_image():
    upload_file = st.file_uploader(label='Pick an image to upload')
    if upload_file is not None:
        image_data = upload_file.getvalue()
        st.image(image_data)
        return PIL.Image.open(io.BytesIO(image_data)).convert('RGB')
    return None


def predict(image):
    model = load_model()
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        class_idx = torch.argmax(probs).item()
    st.success(f"Result: **{labels[class_idx]}** skin")
    st.write(f"Confidence: {probs[class_idx].item() * 100:.1f}%")


def main():
    st.title("Acne Classifier")
    image = load_image()
    if st.button('Predict'):
        if image is not None:
            st.write('Calculating results...')
            predict(image)
        else:
            st.warning('Please upload an image first.')


if __name__ == '__main__':
    main()