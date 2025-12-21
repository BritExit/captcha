import torch
from torchvision.transforms import ToTensor
from Testmodel import CNN
from PIL import Image
import pandas as pd
import os

alphabet = (
    [str(i) for i in range(10)] +
    [chr(i) for i in range(65, 91)]
)
alphabet = ''.join(alphabet)

def predict_image(model, img_path):
    img = Image.open(img_path).convert("RGB")
    img = ToTensor()(img).unsqueeze(0)

    if torch.cuda.is_available():
        img = img.cuda()

    model.eval()
    with torch.no_grad():
        char_out, color_out = model(img)

    char_out = char_out.view(5, 36)
    color_out = color_out.view(5, 2)

    chars = []
    for i in range(5):
        ch = alphabet[torch.argmax(char_out[i]).item()]
        is_red = torch.argmax(color_out[i]).item() == 0  # r=[1,0]

        if is_red:
            chars.append(ch)

    return ''.join(chars)


def generate_submission():
    model = CNN()
    model.load_state_dict(torch.load("./checkpoints/model.pth"))
    if torch.cuda.is_available():
        model = model.cuda()

    test_dir = "./dataset/test/images"
    ids = sorted(os.listdir(test_dir))

    rows = []
    for imgname in ids:
        label = predict_image(model, os.path.join(test_dir, imgname))
        rows.append([imgname, label])

    df = pd.DataFrame(rows, columns=["id", "label"])
    df.to_csv("submission.csv", index=False)
    print("submission.csv 已生成！")


if __name__ == "__main__":
    generate_submission()
