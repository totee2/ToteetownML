
# In[23]:

from transformers import AutoFeatureExtractor, AutoModel
import numpy
import sklearn


model_ckpt = "google/vit-base-patch16-224"
extractor = AutoFeatureExtractor.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)


def createFullImagePath(imageName: str):
    return "../toteetown.backend/assets/images/" + imageName

import torchvision.transforms as T


# Data transformation chain.
transformation_chain = T.Compose(
    [
        # We first resize the input image to 256x256 and then we take center crop.
        T.Resize(int((256 / 224) * extractor.size["height"])),
        T.CenterCrop(extractor.size["height"]),
        T.ToTensor(),
        T.Normalize(mean=extractor.image_mean, std=extractor.image_std),
    ]
)

import torch
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def extract_embeddings(model: torch.nn.Module):
    """Utility to compute embeddings."""
    device = model.device

    def pp(batch):
        
        #images = batch["image"]
        imagePath = createFullImagePath(batch["imageName"])
        print(imagePath)
        images = [Image.open(imagePath).convert('RGB')]
        
        # `transformation_chain` is a compostion of preprocessing
        # transformations we apply to the input images to prepare them
        # for the model. For more details, check out the accompanying Colab Notebook.
        image_transformed = torch.stack(
            [transformation_chain(image) for image in images])
        new_image = {"pixel_values": image_transformed.to(device)}
        with torch.no_grad():
            embeddings = model(**new_image).last_hidden_state[:, 0].cpu()
        return embeddings

    return pp


device = "cuda" if torch.cuda.is_available() else "cpu"
extract_fn = extract_embeddings(model.to(device))


import json
with open('../toteetown.backend/dogInfo.json') as f:
   data = json.load(f)

print(len(data))
candidate_embeddings = list(map(extract_fn,data))


print(len(candidate_embeddings))


print(candidate_embeddings[0].size())


# In[23]:

embeddings_scikit = numpy.array(list(map(lambda obj: obj.numpy(), candidate_embeddings)))

#embeddings_scikit = numpy.array([[]])
#for embedding in candidate_embeddings:
 #   embeddings_scikit = numpy.append (embeddings_scikit, embedding.numpy(), axis=0)
    
print(embeddings_scikit.shape)

print(embeddings_scikit[0].ndim)

nsamples, nx, ny = embeddings_scikit.shape

d2_scikit = embeddings_scikit.reshape((nsamples,nx*ny))

from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=30, algorithm='ball_tree').fit(d2_scikit)


distances, indices = nbrs.kneighbors(d2_scikit)
indices



print(indices[0])

numpy.savetxt("../toteetown.backend/neighbours.txt", indices, fmt='%d')
