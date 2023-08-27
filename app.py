import streamlit as st
import pandas as pd
import clip
import torch
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import PIL
from pathlib import Path

import os
import clip
import torch
from torch.utils.data import Dataset, DataLoader
import PIL
import pickle
from tqdm import tqdm


class Images(Dataset):
        """Images dataset"""
        
        def __init__(self, image_list, transform):
            """
            Args:
                image_list: List of image paths.
                transform : Transform to be applied on a sample.
            """
            self.image_list = image_list
            self.transform = transform
        
        def __len__(self):
            return len(self.image_list)
        
        def __getitem__(self, idx):
            image_path = self.image_list[idx]
            image = PIL.Image.open(image_path)
            image = self.transform(image)
            data = {'image':image, 
                    'img_path': image_path}
            return data




device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device, jit=True)



# load embeddings from file

st.header('Search for your picture')

folder_path = st.text_input('Enter Directory Path: ',key=1)

# folder_path= Path(folder_path)
# print(folder_path)
if folder_path is not '':
    image_list = [folder_path+file for file in os.listdir(folder_path)]
    cleaned_image_list = []
    for image_path in image_list:
        try:
            PIL.Image.open(image_path)
            cleaned_image_list.append(image_path)
        except:
            # print(f"Failed for {image_path}")
            continue

    dataset = Images(cleaned_image_list,preprocess)

    dataloader = DataLoader(dataset, 
                            batch_size=256,
                            shuffle=True)    
    image_paths = []
    embeddings = []
    for data in dataloader:
        with torch.no_grad():
            X = data['image'].to(device)
            image_embedding = model.encode_image(X)
            img_path = data['img_path']
            image_paths.extend(img_path)
            embeddings.extend([torch.Tensor(x).unsqueeze(0).cpu() for x in image_embedding.tolist()])

    image_embeddings = dict(zip(image_paths,embeddings))

    # save to pickle file for the app
    # print("Saving image embeddings")

    search_term = 'a picture of ' + st.text_input('Picture to be searched: ',key=2)
    
    search_embedding = model.encode_text(clip.tokenize(search_term).to(device)).cpu().detach().numpy()

    st.sidebar.header('Settings')
    top_number = st.sidebar.slider('Number of Search Results', min_value=3, max_value=len(image_embeddings))
    picture_width = st.sidebar.slider('Display Width', min_value=100, max_value=500)

    df_rank = pd.DataFrame(columns=['image_path','sim_score'])

    for path,embedding in image_embeddings.items():
        sim = cosine_similarity(embedding,
                                search_embedding).flatten().item()
        df_rank = pd.concat([df_rank,pd.DataFrame(data=[[path,sim]],columns=['image_path','sim_score'])])
    df_rank.reset_index(inplace=True,drop=True)

    df_rank.sort_values(by='sim_score',
                        ascending=False,
                        inplace=True,
                        ignore_index=True)


    col1, col2, col3 = st.columns(3)

    df_result = df_rank.head(top_number)

    for i in range(top_number):
        
        if i % 3 == 0:
            with col1:
                st.image(PIL.Image.open(df_result.loc[i,'image_path']),width=picture_width)
        elif i % 3 == 1:
            with col2:
                st.image(PIL.Image.open(df_result.loc[i,'image_path']),width=picture_width)
        elif i % 3 == 2:
            with col3:
                st.image(PIL.Image.open(df_result.loc[i,'image_path']),width=picture_width)