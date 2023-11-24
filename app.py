import os
import streamlit as st
import numpy as np
import cv2  # opencv-python-headless

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import models, transforms

from utils import show_icon


def plot_perturbed_image(uploaded_file, sign_data_grad):
    noise = sign_data_grad.squeeze(0).permute(1, 2, 0).numpy()

    noise_image = (noise * 255).astype(np.uint8)
    cv2.imwrite(f'./pic/{uploaded_file}_noise_image.png', noise_image)


def plot_gradinet_distribution(uploaded_file, sign_data_grad):
    gradients = sign_data_grad.squeeze(0).permute(1, 2, 0).numpy()

    plt.figure()
    plt.imshow(gradients, cmap='hot', interpolation='nearest')
    plt.title('Gradient Direction')
    plt.axis('off')
    plt.colorbar()
    plt.savefig(f'./pic/{uploaded_file}_gradient_distribution.png')


def plot_attacked_image(uploaded_file, attacked_image):
    attacked_image_detach = attacked_image.detach()
    attacked_image_detach = attacked_image_detach.squeeze(0).permute(1, 2, 0).numpy()

    attacked_image = (attacked_image_detach * 255).astype(np.uint8)
    cv2.imwrite(f'./pic/{uploaded_file}_attacked_image.png', attacked_image)


def fgsm_attack(image, epsilon, data_grad, uploaded_file):
    sign_data_grad = data_grad.sign()  # [1, 3. 224, 224]

    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    plot_gradinet_distribution(uploaded_file, sign_data_grad)
    plot_perturbed_image(uploaded_file, sign_data_grad)
    plot_attacked_image(uploaded_file, perturbed_image)

    return perturbed_image


def plot_difference(output, attack_output, uploaded_file):

    class_list = get_class_idx()

    output = F.softmax(output.detach(), dim=1).squeeze()
    attack_output = F.softmax(attack_output.detach(), dim=1).squeeze()

    output_topk_values, output_topk_indices = torch.topk(output, k=10)
    attack_output_topk_values, attack_output_topk_indices = torch.topk(attack_output, k=10)

    output_topk_classes = [class_list[i] for i in output_topk_indices]
    attack_output_topk_classes = [class_list[i] for i in attack_output_topk_indices]

    # 可视化差值较大的前十个数据
    plt.figure()
    plt.bar(output_topk_classes, output_topk_values.numpy(), alpha=0.5, label='Original Output')
    plt.bar(attack_output_topk_classes, attack_output_topk_values.numpy(), alpha=0.5, label='Attacked Output')
    plt.xticks(rotation=90)
    plt.xlabel('Category')
    plt.ylabel('Probability')
    plt.title('Probability Distributions')
    plt.legend()
    plt.savefig(f'./pic/{uploaded_file}_difference.png', bbox_inches='tight')


def image_attack(uploaded_file, epsilon):

    class_list = get_class_idx()

    model = models.efficientnet_b7(pretrained=True, num_classes=1000)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(f"./pic/{uploaded_file}")
    image = transform(image).unsqueeze(0)

    image.requires_grad = True
    output = model(image)
    target_class = torch.argmax(output, dim=1).item()

    orginal_class = class_list[target_class]

    loss = nn.CrossEntropyLoss()
    gradients = torch.autograd.grad(loss(output, torch.tensor(target_class).reshape(-1)), image)[0]

    perturbed_image = fgsm_attack(image, epsilon, gradients, uploaded_file.split('.')[0])

    attack_output = model(perturbed_image)
    attack_pred = torch.argmax(attack_output, dim=1).item()

    attack_class = class_list[attack_pred]

    plot_difference(output, attack_output, uploaded_file.split('.')[0])

    return orginal_class, attack_class


def get_class_idx(response='imagenet_labels.txt'):
    with open(response, 'r') as f:
        content = f.read()
        class_list = list(content.replace(',', ' ').split('\n'))[:-1]
        class_labels = [label[1:-2] for label in class_list]

        return class_labels


def main():
    st.set_page_config(page_title='FGSM Visualizer', page_icon=":dart:")

    show_icon(":bow_and_arrow:")
    st.markdown("#  :rainbow[Your Own Attacker]")

    uploaded_file = st.file_uploader("### Upload an image", type=["png", "jpg"])

    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()

        with open(f"./pic/{uploaded_file.name}", 'wb') as file:
            file.write(bytes_data)

        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        with st.expander(f'Choose the proper parameter of FGSM'):
            epsilon = st.number_input('perturbing rate', min_value=0.001, max_value=1.000, step=0.001, format='%.3f', )

        original_image_class, attacked_image_class = image_attack(uploaded_file.name, epsilon)

        try:
            st.write(f"The class of the uploaded image is :point_right: :rainbow[{original_image_class}!]")
            st.divider()

            st.image(f"./pic/{uploaded_file.name.split('.')[0]}_attacked_image.png", caption="Attacked Image", use_column_width=True)
            st.write(f"The class of the attacked image is :poop: :rainbow[{attacked_image_class}]")
            st.divider()

            st.image(f"./pic/{uploaded_file.name.split('.')[0]}_gradient_distribution.png", caption="Gradient Distribution", use_column_width=True)
            st.write(f":earth_africa: The gradient distribution of uploaded image.")
            st.divider()

            st.image(f"./pic/{uploaded_file.name.split('.')[0]}_noise_image.png", caption="Noise Image", use_column_width=True)
            st.write(f":: The noise image after attack.")
            st.divider()

            st.image(f"./pic/{uploaded_file.name.split('.')[0]}_difference.png", caption="Output Difference", use_column_width=True)
            st.write(f"The output distribution before attack and after attack :eyes: ")

        except:
            st.write(":monkey: Error occured, please try again.")

    st.divider()
    st.markdown(
        """
        ---
        Follow me on:

        Github → [@Klayand](https://Klayand.github.io) :dragon_face:

        Google Scholar → [@Zikai Zhou](https://scholar.google.com/citations?user=u6TjscAAAAAJ) :eyes:

        """
    )


if __name__ == "__main__":
    main()

