from PIL import Image
import numpy as np

import torch
import torch.nn
import torch.optim as optim
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from torchvision.models import vgg19, VGG19_Weights
from tqdm import tqdm
import StyleNet
import utils
import clip
import torch.nn.functional as F
from template import imagenet_templates

from PIL import Image 
import PIL 
from torchvision import utils as vutils
import argparse
from torchvision.transforms.functional import adjust_contrast

import style_retrieval
from utils import slerp, seed_everything
parser = argparse.ArgumentParser()

parser.add_argument('--content_path', type=str, default="./face.jpg",
                    help='Image resolution')
parser.add_argument('--content_name', type=str, default="face",
                    help='Image resolution')
parser.add_argument('--exp_name', type=str, default="exp1",
                    help='Image resolution')
parser.add_argument('--text', type=str, default="Fire",
                    help='Image resolution')
parser.add_argument('--lambda_tv', type=float, default=2e-3,
                    help='total variation loss parameter')
parser.add_argument('--lambda_patch', type=float, default=9000,
                    help='PatchCLIP loss parameter')
parser.add_argument('--lambda_dir', type=float, default=500,
                    help='directional loss parameter')
parser.add_argument('--lambda_c', type=float, default=150,
                    help='content loss parameter')
parser.add_argument('--crop_size', type=int, default=128,
                    help='cropped image size')
parser.add_argument('--num_crops', type=int, default=64,
                    help='number of patches')
parser.add_argument('--img_width', type=int, default=512,
                    help='size of images')
parser.add_argument('--img_height', type=int, default=512,
                    help='size of images')
parser.add_argument('--max_step', type=int, default=200,
                    help='Number of domains')
parser.add_argument('--lr', type=float, default=5e-4,
                    help='Number of domains')
parser.add_argument('--thresh', type=float, default=0.7,
                    help='Number of domains')
parser.add_argument('--n_samples', type=int, default=3, 
                    help='Number of samples')
parser.add_argument('--device', type=str, default="cuda", 
                    help='device to run the code')

args = parser.parse_args()
print(args)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device(args.device)
assert (args.img_width%8)==0, "width must be multiple of 8"
assert (args.img_height%8)==0, "height must be multiple of 8"

VGG = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
VGG.to(device)

for parameter in VGG.parameters():
    parameter.requires_grad_(False)
    
def img_denormalize(image):
    mean=torch.tensor([0.485, 0.456, 0.406]).to(device)
    std=torch.tensor([0.229, 0.224, 0.225]).to(device)
    mean = mean.view(1,-1,1,1)
    std = std.view(1,-1,1,1)

    image = image*std +mean
    return image

def img_normalize(image):
    mean=torch.tensor([0.485, 0.456, 0.406]).to(device)
    std=torch.tensor([0.229, 0.224, 0.225]).to(device)
    mean = mean.view(1,-1,1,1)
    std = std.view(1,-1,1,1)

    image = (image-mean)/std
    return image

def clip_normalize(image,device):
    image = F.interpolate(image,size=224,mode='bicubic').to(device)
    mean=torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
    std=torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)
    mean = mean.view(1,-1,1,1)
    std = std.view(1,-1,1,1)

    image = (image-mean)/std
    return image

    
def get_image_prior_losses(inputs_jit):
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    
    return loss_var_l2

def compose_text_with_templates(text: str, templates=imagenet_templates) -> list:
    return [template.format(text) for template in templates]

def compose_img(prompt, limit=5):
    table = style_retrieval.init_dataset()
    return style_retrieval.search(table, prompt, limit)



content_path = args.content_path
content_image = utils.load_image2(content_path, img_height=args.img_height,img_width=args.img_width)
content = args.content_name
exp = args.exp_name

content_image = content_image.to(device)

content_features = utils.get_features(img_normalize(content_image), VGG)



output_image = content_image


cropper = transforms.Compose([
    transforms.RandomCrop(args.crop_size)
])
augment = transforms.Compose([
    transforms.RandomPerspective(fill=0, p=1,distortion_scale=0.5),
    transforms.Resize(224)
])

clip_model, preprocess = clip.load('ViT-B/32', device, jit=False)

prompt = args.text

source = "a Photo"

with torch.no_grad():
    rs = compose_img(prompt)
    style_img_features = None
    for i in range(len(rs)):
        retireval_image = utils.load_image2(rs[i]['image_uri'], img_height=args.img_height, img_width=args.img_width)
        if i == 0:
            style_img_features = clip_model.encode_image(clip_normalize(retireval_image, device))
        else:
            style_img_features += clip_model.encode_image(clip_normalize(retireval_image, device))
            
    style_img_features /= len(rs)
    style_img_features /= style_img_features.norm(dim=-1, keepdim=True)
    
    # style img's content embedding
    template_text = compose_text_with_templates(prompt, imagenet_templates)
    tokens = clip.tokenize(template_text).to(device)
    style_content_features = clip_model.encode_text(tokens).detach()
    style_content_features = style_content_features.mean(axis=0, keepdim=True)
    style_content_features /= style_content_features.norm(dim=-1, keepdim=True)
    
    
    template_source = compose_text_with_templates(source, imagenet_templates)
    tokens_source = clip.tokenize(template_source).to(device)
    text_source = clip_model.encode_text(tokens_source).detach()
    text_source = text_source.mean(axis=0, keepdim=True)
    text_source /= text_source.norm(dim=-1, keepdim=True)
    source_features = clip_model.encode_image(clip_normalize(content_image,device))
    source_features /= (source_features.clone().norm(dim=-1, keepdim=True))

intp = []
for sample in range(0, args.n_samples):
    seed_everything(900131)
    style_net = StyleNet.UNet()
    style_net.to(device)

    style_weights = {'conv1_1': 0.1,
                    'conv2_1': 0.2,
                    'conv3_1': 0.4,
                    'conv4_1': 0.8,
                    'conv5_1': 1.6}

    content_weight = args.lambda_c

    optimizer = optim.Adam(style_net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    steps = args.max_step

    content_loss_epoch = []
    style_loss_epoch = []
    total_loss_epoch = []
    print(f"Sample {sample}/{args.n_samples - 1}")
    text_features = slerp(style_img_features, style_content_features, sample / (args.n_samples - 1))
    
    num_crops = args.num_crops
    progress = tqdm(range(0, steps+1))
    for epoch in progress:
        scheduler.step()
        target = style_net(content_image, use_sigmoid=True).to(device)
        target.requires_grad_(True)
        
        target_features = utils.get_features(img_normalize(target), VGG)
        
        content_loss = 0

        content_loss += torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
        content_loss += torch.mean((target_features['conv5_2'] - content_features['conv5_2']) ** 2)

        loss_patch=0 
        img_proc =[]
        for n in range(num_crops):
            target_crop = cropper(target)
            target_crop = augment(target_crop)
            img_proc.append(target_crop)

        img_proc = torch.cat(img_proc,dim=0)
        img_aug = img_proc

        image_features = clip_model.encode_image(clip_normalize(img_aug,device))
        image_features /= (image_features.clone().norm(dim=-1, keepdim=True))
        
        img_direction = (image_features-source_features)
        img_direction /= img_direction.clone().norm(dim=-1, keepdim=True)
        
        text_direction = (text_features-text_source).repeat(image_features.size(0),1)
        text_direction /= text_direction.norm(dim=-1, keepdim=True)
        loss_temp = (1 - torch.cosine_similarity(img_direction, text_direction, dim=1))
        loss_temp[loss_temp<args.thresh] =0
        loss_patch+=loss_temp.mean()
        
        glob_features = clip_model.encode_image(clip_normalize(target,device))
        glob_features /= (glob_features.clone().norm(dim=-1, keepdim=True))
        
        glob_direction = (glob_features-source_features)
        glob_direction /= glob_direction.clone().norm(dim=-1, keepdim=True)
        
        loss_glob = (1- torch.cosine_similarity(glob_direction, text_direction, dim=1)).mean()
        
        reg_tv = args.lambda_tv*get_image_prior_losses(target)

        total_loss = args.lambda_patch*loss_patch + content_weight * content_loss+ reg_tv+ args.lambda_dir*loss_glob
        total_loss_epoch.append(total_loss)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        progress.set_description(f'loss:{total_loss.item():.4f}, content:{content_loss.item():.4f}, patch:{loss_patch.item():.4f}, dir:{loss_glob.item():.4f}, tv:{reg_tv.item():.4f}')
    output_image = target.clone()
    output_image = torch.clamp(output_image,0,1)
    output_image = adjust_contrast(output_image,1.5)
    intp.append(output_image)

output_image = torch.stack(intp, dim=1)
prompt = prompt.replace(' ', '_')
output_name = f'./demo-intp/{content}_{prompt}.png'
output_image = make_grid(output_image, nrow=args.n_samples)
save_image(output_image, output_name)