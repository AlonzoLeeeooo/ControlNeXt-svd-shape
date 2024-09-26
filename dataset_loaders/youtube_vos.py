import os
import random
import numpy as np
from PIL import Image
import cv2
import torch 
from torch.utils.data import Dataset

import json

# load util functions
from utils.data_utils import center_crop_and_resize, image_to_tensor, binarize_tensor


def make_train_dataset(args, mode='ubc', **kwargs):
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_type == 'webvid':
        from utils.vid_dataset import WebVid10M
        dataset = WebVid10M(args.csv_path,
                            args.video_folder,
                            args.condition_folder,
                            args.motion_folder)
    elif args.dataset_type == 'youtube_vos':
        dataset_configs = {}
        dataset_configs['annotation_path'] = kwargs['annotation_path']
        dataset_configs['tokenizer'] = kwargs['tokenizer']
        dataset_configs['feature_extractor'] = kwargs['feature_extractor']
        dataset_configs['image_processor'] = kwargs['image_processor']
        dataset = YouTubeDataset(
            video_path=args.video_path,
            caption_path=args.caption_path,
            shape_path=args.shape_path,
            **dataset_configs
        )
    elif args.dataset_type == 'ubc':
        from utils.vid_dataset import UBCFashion
        dataset = UBCFashion(
            args.meta_info_path,
            args.width,
            args.height,
            args.sample_n_frames,
            args.interval_frame,
            stage=args.train_stage,
            ref_aug=args.ref_augment)
    return dataset


class YouTubeDataset(Dataset):
    def __init__(
        self,
        width=512,
        height=512,
        n_sample_frames=14,
        output_fps=3,
        video_path=None,
        caption_path=None,
        shape_path=None,
        use_empty_prompt=False,
        model_name=None,
        **kwargs
    ):

        self.width = width
        self.height = height

        self.model_name = model_name
        self.n_sample_frames = n_sample_frames
        self.output_fps = output_fps 
        self.use_empty_prompt = use_empty_prompt
        
        self.video_path = video_path
        self.caption_path = caption_path
        with open(self.caption_path, 'r') as f:
            self.video_caption_dict = json.load(f)
        
        self.shape_path = shape_path
        self.video_files = os.listdir(video_path)
        
        # we use fixed seed here (optional)
        random.seed(42)
        random.shuffle(self.video_files)

        # Load meta data
        with open(kwargs['annotation_path'], 'r') as f:
            self.meta_data = json.load(f)

        
        # please note that we need to set backbone-specific parameters here
        # these parameters are used in the forward function
        self.tokenizer = kwargs['tokenizer'] 
        self.feature_extractor = kwargs['feature_extractor']
        self.image_processor = kwargs['image_processor']


    def __len__(self):
        return len(self.video_files)

        
    def __getitem__(self, index):
        selected_folder = self.video_files[index]
        seg_pixel = (list(self.meta_data['videos'][selected_folder]['objects'].keys()))
        selected_seg_pixel = random.choice(seg_pixel)

        # Load in video frames
        frame_path_list = sorted(os.listdir(os.path.join(self.video_path, selected_folder)))
        pil_frames = []
        tensor_frames = []
        for frame_path in frame_path_list:
            frame = Image.open(os.path.join(self.video_path, selected_folder, frame_path))
            frame = frame.resize((self.width, self.height))
            pil_frames.append(frame)
            frame = center_crop_and_resize(frame, (self.width, self.height))
            frame = image_to_tensor(frame)
            tensor_frames.append(frame)
        
        # Sample index
        max_start_index = len(frame_path_list) - self.n_sample_frames
        start_index = random.randint(0, max_start_index)
        end_index = start_index + self.n_sample_frames
        tensor_frames = tensor_frames[start_index:end_index]
        
        pixel_values = torch.stack(tensor_frames)
        reference_pixel_values = pixel_values[0]        

        # Load shape annotation
        shape_path_list = sorted(os.listdir(os.path.join(self.shape_path, selected_folder)))
        nparray_shapes = []
        for shape_path in shape_path_list[start_index:end_index]:
            shape = Image.open(os.path.join(self.shape_path, selected_folder, shape_path))
            shape = np.array(shape)
            shape = cv2.resize(shape, (self.width, self.height))
            init_shape = np.zeros((shape.shape))
            indices = np.where(shape == int(selected_seg_pixel))
            init_shape[indices[0], indices[1]] = 1
            nparray_shapes.append(init_shape)
       
        tensor_shapes = []
        for nparray_shape in nparray_shapes:
            nparray_shape = torch.from_numpy(nparray_shape.astype(np.float32)).unsqueeze(0)
            nparray_shape = binarize_tensor(nparray_shape)
            tensor_shapes.append(nparray_shape)
        tensor_shapes = torch.stack(tensor_shapes, dim=0)
        tensor_shapes = torch.cat([tensor_shapes] * 3, dim=1)


        return {
            "pixel_values": pixel_values,
            "guide_values": tensor_shapes,
            "reference_image": reference_pixel_values
            }
            
from transformers import CLIPTokenizer, CLIPImageProcessor
from diffusers.image_processor import VaeImageProcessor
def main():
    # Define the tokenizer, feature_extractor, and image_processor
    tokenizer = CLIPTokenizer.from_pretrained("/home/dongeliugroup/liuchang2/checkpoints/i2vgenxl", subfolder='tokenizer')
    feature_extractor = CLIPImageProcessor.from_pretrained("/home/dongeliugroup/liuchang2/checkpoints/i2vgenxl", subfolder='feature_extractor')
    image_processor = VaeImageProcessor(vae_scale_factor=1, do_resize=False)

    dataset = YouTubeDataset(
        video_path='/home/dongeliugroup/liuchang2/data/youtube-vos/train_zip/train/train/JPEGImages',
        caption_path='/home/dongeliugroup/liuchang2/data/youtube-vos/video-caption-dict.json',
        shape_path='/home/dongeliugroup/liuchang2/data/youtube-vos/train_zip/train/train/Annotations',
        annotation_path='/home/dongeliugroup/liuchang2/data/youtube-vos/train_zip/train/train/meta.json',
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        image_processor=image_processor,
        use_bounding_box=True,
        pred_shape=True
    )

    for i in range(len(dataset)):
        data = dataset[i]
        print(f"Processed video {i+1}/{len(dataset)}")

if __name__ == "__main__":
    main()

