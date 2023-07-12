import torch
import gradio as gr
from torchvision import transforms


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def demo() :
    loaded_model = torch.jit.load('model_vit.script.pt')


    def predict(image):
        image = transforms.ToTensor()(image.convert('RGB')).unsqueeze(0)
        if image is None:
            return None
        # image = torch.tensor(image[None, None, ...], dtype=torch.float32)
        preds = loaded_model.forward_jit(image)
        preds = preds[0].tolist()
        return {classes[i]: preds[i] for i in range(10)}

    demo = gr.Interface(fn=predict, # mapping function from input to output
                    inputs=gr.Image(type="pil"), # what are the inputs?
                    outputs=[gr.Label(num_top_classes=10, label="Predictions")],
                    title='EMLO-ViT - 🐱 🐶',
                    description='ViT trained on CIFAR10',
                    article='For EMLO-S8',
                    )
    
    demo.launch(server_port=8080, server_name='0.0.0.0')
    

def main():
    demo()

if __name__ == "__main__":
    main()