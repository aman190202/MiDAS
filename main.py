import cv2
import torch
import urllib.request
import matplotlib.pyplot as plt
from tempfile import TemporaryDirectory
import os 

des_file = input("Enter the location of the desired file : ")

# MiDas Files are loaded
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
urllib.request.urlretrieve(url, filename)
model_type = "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform


with TemporaryDirectory(prefix="myapp-") as location:


    print(location)

    #Module1
    print("--Video Fragmentation begins--")
    currentframe = 0
    cam = cv2.VideoCapture(des_file) 
    while(True):
        ret,frame = cam.read()
        if ret:
            name =str(currentframe) + '.jpg'
            name = os.path.join(location,name)
            cv2.imwrite(name, frame)
            currentframe += 1
        else:
            break
    cam.release()
    cv2.destroyAllWindows()
    print("--Video Fragmentation Ends--")

    #Module2
    for filename in sorted(os.listdir(location)):

        f = os.path.join(location, filename)

        if os.path.isfile(f):
            img = cv2.imread(f)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            input_batch = transform(img).to(device)
            with torch.no_grad():
                prediction = midas(input_batch)
                prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
                ).squeeze()

        output = prediction.cpu().numpy()
        plt.imshow(output)
        plt.axis('off')
        plt.savefig(f)
        plt.close()

    #Module3
    img_array = []
    dir_len=len([name for name in os.listdir(location) if os.path.isfile(os.path.join(location, name))])
    for i in range(0,dir_len):
        img = cv2.imread(location+'/'+str(i)+'.jpg')
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
    out = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 30, size )
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

    
'''
@article{Ranftl2020,
	author    = {Ren\'{e} Ranftl and Katrin Lasinger and David Hafner and Konrad Schindler and Vladlen Koltun},
	title     = {Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer},
	journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
	year      = {2020},
}

@article{Ranftl2021,
	author    = {Ren\'{e} Ranftl and Alexey Bochkovskiy and Vladlen Koltun},
	title     = {Vision Transformers for Dense Prediction},
	journal   = {ArXiv preprint},
	year      = {2021},
}

'''
