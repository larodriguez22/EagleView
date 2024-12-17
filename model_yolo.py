#%%

# model.save('yolo_model')
from ultralytics import YOLO
import os

curr_path = os.getcwd()
import tkinter
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
# matplotlib.use('TkAgg')

# %matplotlib inline

#test_image=os.path.join("juan-pablo-garcia-marruecosjpg.jpeg")
#test_image=os.path.join("descarga_3.webp")
#test_image=os.path.join("WhatsApp Image 2023-11-27 at 19.01.11.jpeg")
#test_image=os.path.join("download.jpeg")

#test_image

from ultralytics import YOLO
import os

curr_path = os.getcwd()
import tkinter
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


def get_prediction(image_path, model_path, otro):
    if otro == 0:
        model2 = YOLO(model_path)
        img = plt.imread(image_path)
        imgplot = plt.imshow(img)
        plt.show()
        res = model2(image_path)
        res_plotted = res[0].plot()
        plt.imshow(res_plotted)
        plt.title("Image with predictions", fontsize=40)
        plt.xticks([])
        plt.yticks([])
        plt.savefig("image.png")
        plt.show()

        return res_plotted
    # model2 = YOLO(model_path)  # load a custom model
    else:
        model2 = load_model(model_path)

        # Perform inference with the loaded model
        img_array = plt.imread(image_path)
        img_array = img_array.astype('float32') / 255.0  # Normalize image
        img_array = img_array.reshape(
            (1, ) + img_array.shape)  # Reshape for model input
        res = model2.predict(img_array)

        # Plot predictions on the image
        plt.imshow(img)
        plt.title("Image with predictions", fontsize=40)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(res[0])  # Assuming res[0] contains the predicted image
        plt.savefig("image.png")
        plt.show()

        return res[0]


# def get_prediction(image_path):
#     model2 = YOLO('train47/weights/best.pt')  # load a custom model
#     img = plt.imread(image_path)
#     imgplot = plt.imshow(img)
#     plt.show()
#     res = model2(image_path)
#     res_plotted = res[0].plot()
#     plt.imshow(res_plotted)
#     plt.title("Image with predictions", fontsize = 40)
#     plt.xticks([])
#     plt.yticks([])
#     plt.show()
#     return res_plotted

# test_image=os.path.join("download.jpeg")
# get_prediction(test_image)
#
##%%
#
## Predict
#res = model2(test_image)
#res_plotted = res[0].plot()
#
## Display image with predictions
#plt.imshow(res_plotted)
#plt.title("Image with predictions", fontsize = 40)
#plt.xticks([])
#plt.yticks([])
