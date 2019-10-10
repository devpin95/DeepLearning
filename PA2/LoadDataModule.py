# coding: utf-8

# In[70]:


from zipfile import ZipFile
import numpy as np
import matplotlib.pyplot as plt

'''load your data here'''


class LoadDataModule(object):
    def __init__(self):
        self.DIR = './'
        pass

    # Returns images and labels corresponding for training and testing. Default mode is train.
    # For retrieving test data pass mode as 'test' in function call.
    def load(self, mode='train'):
        label_filename = mode + '_labels'
        image_filename = mode + '_images'
        label_zip = self.DIR + label_filename + '.zip'
        image_zip = self.DIR + image_filename + '.zip'
        with ZipFile(label_zip, 'r') as lblzip:
            labels = np.frombuffer(lblzip.read(label_filename), dtype=np.uint8, offset=8)
        with ZipFile(image_zip, 'r') as imgzip:
            images = np.frombuffer(imgzip.read(image_filename), dtype=np.uint8, offset=16).reshape(len(labels), 784)
        return images, labels


# In[71]:
# Usage

ld = LoadDataModule()

# In[78]:


# Now let's load the dataset
images, labels = ld.load('train')

# In[73]:


images.shape

# In[74]:


labels.shape

# In[75]:


labels[0]  # Label of first image

# In[76]:


images[0, :]  # First image

# In[77]:


# Plot the first image
plt.imshow(np.reshape(images[0, :], (28, 28)))