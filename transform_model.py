import sys
sys.path.insert(0, "./model")

import argparse
import time
import glob
from torch.autograd import Variable
import torch
from utils.utils import *
from AtJ_model import AtJ

print("*****************************************\n"
      "This is the test code for NTIRE19-Dehaze\n"
      "Pre-request: python-3.6 and pytorch-1.0\n"
      "*****************************************\n"
      "Note: the original model is trained on GPU.\n"
      "The conversion between CPU and GPU model can generate the precesion error.\n"
      "Please keep the original setups and evaluate the network on GPU to reproduce the exact results.\n"
      "Please ignore UserWarning if there is any, which is caused by pytorch updating issues.")

parser = argparse.ArgumentParser(description="Pytorch AtJ_model Evaluation")
parser.add_argument("--cuda", default=True, action="store_true", help="use cuda? Default is True")
parser.add_argument("--model", type=str, default="AtJ_model", help="model path")
parser.add_argument("--test", type=str, default="mydehaze", help="testset path")
opt = parser.parse_args()
cuda = opt.cuda
device_label = 'GPU' if opt.cuda else 'CPU'

if cuda and not torch.cuda.is_available():
    raise Exception(">>No GPU found, please run without --cuda")

if not cuda:
    print(">>Run on *CPU*, the running time will be longer than reported GPU run time. \n"
          ">>To run on GPU, please run the script with --cuda option")

save_path = 'result_{}_{}'.format(opt.model,device_label)
save_path = "./result/"
checkdirctexist(save_path)

model = AtJ()
model_path = "model/AtJ_model.pth"
state_dict = torch.load(model_path)['model'].state_dict()
model.load_state_dict(state_dict)

image_list = glob.glob(os.path.join(opt.test, '*.jpeg'))

print(">>Start testing...\n"
      "\t\t Model: {0}\n"
      "\t\t Test on: {1}\n"
      "\t\t Results save in: {2}".format(opt.model, opt.test, save_path))

avg_elapsed_time = 0.0
count = 0
for image_name in image_list:
    count += 1
    print(">>Processing ./{}".format(image_name))
    im_input, W, H = get_image_for_test(image_name)

    with torch.no_grad():
        im_input = Variable(torch.from_numpy(im_input).float())
        if cuda:
            model = model.cuda()
            model.train(False)
            im_input = im_input.cuda()
        else:
            im_input = im_input.cpu()
            model = model.cpu()
        model.train(False)
        model.eval()
        start_time = time.time()
        # feeding forward
        im_output = model(im_input)
        im_output = im_output[0]
        # compute running time
        elapsed_time = time.time() - start_time
        avg_elapsed_time += elapsed_time

    im_output = im_output.cpu()
    im_output_forsave = get_image_for_save(im_output, W, H)
    path, filename = os.path.split(image_name)
    cv2.imwrite(os.path.join(save_path, filename), im_output_forsave)

print(">>Finished!"
      "It takes average {}s for processing single image on {}\n"
)

