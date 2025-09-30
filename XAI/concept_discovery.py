"""ACE library.

Library for discovering and testing concept activation vectors. It contains
ConceptDiscovery class that is able to discover the concepts belonging to one
of the possible classification labels of the classification task of a network
and calculate each concept's TCAV score..
"""
import os
import pickle
import csv
import numpy as np
from PIL import Image
from pyparsing import original_text_for
import scipy.stats as stats
import skimage.segmentation as segmentation
from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec
import shutil
from sklearn import cluster, metrics, linear_model
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from multiprocessing import dummy as multiprocessing
import os
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
from torchvision import models

from dataloaders.presets import PresetTransform

# This list is used to store the latent activations
# or gradients of the layer specified by the user
layer_activations = []
def forward_hook(module, input, output):
  layer_activations.append(output.squeeze().detach().numpy())
def hook_backwards(module, grad_input, grad_output):
  layer_activations.append(grad_output[0].detach().numpy())

def get_layer_activations(model, layer:str, imgs:np.ndarray, 
                          batch_size:int=50, get_grads_for_cls_id:int=None):
  """Run input images through the PyTorch model and optain activations from
  specified layer

  Args:
    model: PyTorch model
    layer: layer within the model from which the activations are returned
    imgs: numpy-ndarray containing the imgs of shape [num_images, widht, height, channels=3]
    batch_size: batch size used for the calculations (reduces computational effort)
    get_gradients: instead of returning the activations of the specified layer, this parameter
      abuses the function to return the gradients w.r.t. a specific output class

  Returns:
    activations: NumPy array containing the model activations
  """

  if get_grads_for_cls_id is None:
    # register forward hook to get the activations from a specific layer
    hook_handle = model._modules.get(layer)[0].register_forward_hook(forward_hook) ###CHANGE HERE!!!!!
  else:
    # if get_grads_for_cls_id is set, instead register a backward hook which is used
    # to observe the gradients of a backward pass of the specified layer
    hook_handle = model._modules.get(layer)[0].register_full_backward_hook(hook_backwards) ###CHANGE HERE!!!!!
  # clear global list in which layer activations/gradients are stored
  del layer_activations[:]

  # Set the model to evaluation mode
  model.eval()

  # iterate over batches of images in imgs
  for i in tqdm(range(int(imgs.shape[0] / batch_size) + 1), desc='[INFO] calculating activations'):
    # convert images to a PyTorch tensor
    tensor_imgs = torch.tensor(imgs[i * batch_size:(i + 1) * batch_size].transpose((0, 3, 1, 2)), dtype=torch.float32)
    transformed_tensor_imgs = PresetTransform("NCNN").transforms(tensor_imgs)
    transformed_tensor_imgs.to('cuda' if torch.cuda.is_available() else 'cpu')

    # forward pass through the model
    # if get_grads_for_cls_id=None the output is ignored and instead the activations of the 
    # specified layer are observed using a forward hook (see above)
    output = model(transformed_tensor_imgs)

    # if value for get_grads_for_cls_id is passed, calculate the gradients instead of the activations
    # for this, the cross-entropy loss is backpropagated through the network and the gradients are 
    # observed using a backward hook (see above)
    if not get_grads_for_cls_id is None:
      model.zero_grad()
      loss = torch.nn.functional.cross_entropy(
        output,
        torch.zeros(
          output.size(0), output.size(1)
          ).scatter_(
            1, torch.tensor([[get_grads_for_cls_id]] * output.size(0)), 1
            )
      )
      loss.backward()

  # remove the hook
  hook_handle.remove()

  # return the activations/gradients of the specified layer
  return np.concatenate(layer_activations, axis=0)

def create_dirs(working_dir:str, target_class:str, model_name:str, layer:str) -> str:
  """creates directories to load/save cached values and results

  Args:
      working_dir (str): argument passed by the user where the cached values and 
        results should be stored
      target_class (str): argmument passed by the user which class should be 
        interpreted
      model_name (str): argument passed by the user which model should be explained
      layer (str): argument passed by the user which layer should be used to calculate
        the activations

  Returns:
      str: base directory where cached values and results are loaded/saved
  """
  # create working directory if not already exists
  os.makedirs(working_dir, exist_ok=True)
  # create directory for model to be interpreted if not already exists
  os.makedirs(os.path.join(working_dir, model_name), exist_ok=True)
  # create directory for target class the be explained if not already exists
  os.makedirs(os.path.join(working_dir, model_name, target_class), exist_ok=True)
  # create directory for layer to calc acts from if not already exists
  base_dir = os.path.join(working_dir, model_name, target_class, layer)
  os.makedirs(base_dir, exist_ok=True)
  # create directories to load/save cached values and results
  os.makedirs(os.path.join(base_dir, 'acts'), exist_ok=True)
  print(f"[INFO] {os.path.join(os.getcwd(), base_dir)} is used to load/save cached values and results")
  return os.path.normpath(base_dir)

def load_image_from_file(filename:str, shape) -> np.array:
  """Given a filename, try to open the file. If failed, return None.
  Args:
    filename: location of the image file
    shape: the shape of the image file to be scaled
  Returns:
    the image if succeeds, None if fails.
  Rasies:
    exception if the image was not the right shape.
  """
  if not os.path.exists(filename):
    print('Cannot find file: {}'.format(filename))
    return None
  try:
    img = np.array(Image.open(filename).resize(shape, Image.BILINEAR))
    # Normalize pixel values to between 0 and 1.
    img = np.float32(img) / 255.0
    if not (len(img.shape) == 3 and img.shape[2] == 3):
      return None
    else:
      return img
  except Exception as e:
    print(e)
    return None


def load_images_from_files(filenames, max_imgs=500, return_filenames=False,
                           do_shuffle=True, run_parallel=True,
                           shape=(120, 120),
                           num_workers=100):
  """Return image arrays from filenames.
  Args:
    filenames: locations of image files.
    max_imgs: maximum number of images from filenames.
    return_filenames: return the succeeded filenames or not
    do_shuffle: before getting max_imgs files, shuffle the names or not
    run_parallel: get images in parallel or not
    shape: desired shape of the image
    num_workers: number of workers in parallelization.
  Returns:
    image arrays and succeeded filenames if return_filenames=True.
  """
  imgs = []
  # First shuffle a copy of the filenames.
  filenames = filenames[:]
  if do_shuffle:
    np.random.shuffle(filenames)
  if return_filenames:
    final_filenames = []
  if run_parallel:
    pool = multiprocessing.Pool(num_workers)
    imgs = pool.map(lambda filename: load_image_from_file(filename, shape),
                    filenames[:max_imgs])
    if return_filenames:
      final_filenames = [f for i, f in enumerate(filenames[:max_imgs])
                         if imgs[i] is not None]
    imgs = [img for img in imgs if img is not None]
  else:
    for filename in filenames:
      img = load_image_from_file(filename, shape)
      if img is not None:
        imgs.append(img)
        if return_filenames:
          final_filenames.append(filename)
      if len(imgs) >= max_imgs:
        break

  if return_filenames:
    return np.array(imgs), final_filenames
  else:
    return np.array(imgs)

def save_images(addresses, images):
  """Save images in the addresses.

  Args:
    addresses: The list of addresses to save the images as or the address of the
      directory to save all images in. (list or str)
    images: The list of all images in numpy uint8 format.
  """
  if not isinstance(addresses, list):
    image_addresses = []
    for i, image in enumerate(images):
      image_name = '0' * (3 - int(np.log10(i + 1))) + str(i + 1) + '.png'
      image_addresses.append(os.path.join(addresses, image_name))
    addresses = image_addresses
  assert len(addresses) == len(images), 'Invalid number of addresses'
  for address, image in zip(addresses, images):
    Image.fromarray(image).save(address, format='PNG')
  
class ConceptDiscovery(object):
  def __init__(self, target_class:str, layer:str, source_dir:str, base_dir:str, 
               num_random_datasets:int=20, channel_mean:bool=True, input_shape:tuple=(224,224), 
               max_cluster_size:int=50, min_cluster_size:int=30, num_target_class_imgs:int=50, 
               num_workers:int=0, average_image_value:int=117):
    """for a pre-trained classification PyTorch model, the ConceptDiscovery class first
    performs unsupervised concept discovery using images of one class

    Args:
        target_class (str): class of the PyTorch model to be explained
        layer (str): layer within the PyTorch model to calculate the activations from
        source_dir (str): directory where class images as well as random datasets are saved
        base_dir (str): directory to load/save cached values and results
        num_random_datasets (int, optional): number of random datasets (datasets containing 
          random images used as counterparts to calculate CAVs). Defaults to 20.
        channel_mean (bool, optional): whether to mean the channel values of the activations
          before calculating the clusters. Recommended to reduce dimensionality. Note that 
          thi setting is only used for clustering and NOT for calculating the CAVs. Defaults to 
          True.
        input_shape (tuple, optional): image size expected by the PyTorch model. 
          Defaults to (224,224).
        max_cluster_size (int, optional): If cluster exceeds that size, only the 
          top-max_clutser_size images with the minimal cost are kept. Defaults to 50.
        min_cluster_size (int, optional): Cluster that are below this threshold are not 
          considered as concepts. Defaults to 30.
        num_target_class_imgs (int, optional): Number of target class images used to calculate 
          the segmentations from. Defaults to 50.
        num_workers (int, optional): Number of workers used for multiprocessing. Defaults to 0.
        average_image_value (int, optional): ??. Defaults to 117.
    """

    self.target_class = target_class
    self.num_target_class_imgs = num_target_class_imgs
    self.target_class_imgs = None
    # used for saving results and cached values
    self.base_dir = base_dir
    # used for calculating the CAVs
    self.num_random_datasets = num_random_datasets
    # used for calculating the activations
    self.layer = layer
    self.channel_mean = channel_mean
    # used for loading the images from files
    self.source_dir = source_dir
    self.image_shape = input_shape
    # used to store the target class image segmentations
    self.segmentation_dataset = {}
    # used for creating the clusters
    self.max_cluster_size = max_cluster_size
    self.min_cluster_size = min_cluster_size
    # used for multiprocessing
    self.num_workers = num_workers
    # used for creating image patches
    self.average_image_value = average_image_value

  def load_images_from_folder(self, folder_name: str, max_imgs:int=1000):
    """loads images from the sepcified folder name within the folder self.source_dir and saves 
    it to a numpy-ndarray of shape [num_images, self.image_shape[0], self.image_shape[1], 3]

    args:
      folder_name: The name of the concept to be loaded
      max_imgs: maximum number of images to be loaded

    returns:
      numpy-array containing the images """

    folder_path = os.path.join(self.source_dir, folder_name)
    
    imgs_filenames = [
        os.path.join(folder_path, img_path)
        for img_path in os.listdir(folder_path)
    ]

    return load_images_from_files(
        filenames=imgs_filenames,
        max_imgs=max_imgs,
        return_filenames=False,
        do_shuffle=False,
        run_parallel=(self.num_workers > 0),
        shape=self.image_shape,
        num_workers=self.num_workers)

  def create_patches(self, imgs_identifier:str, method:str='slic', 
                     param_dict:dict=None, save:bool=False) -> dict:
    """Creates a set of image patches using superpixel methods.

    This method takes in the concept images and transforms it to a
    dataset made of the patches (segments) of those images.

    Args:
      method: The superpixel method used for creating image patches. One of
        'slic', 'watershed', 'quickshift', 'felzenszwalb'.

      param_dict: Contains parameters of the superpixel method used in the form
                of {'param1':[a,b,...], 'param2':[z,y,x,...], ...}. For instance
                {'n_segments':[15,50,80], 'compactness':[10,10,10]} for slic
                method.
    """
    segmentation_dataset = {}
    
    filepath = os.path.join(
      self.base_dir, f'{imgs_identifier}_image_segmentations.npz'
    )
    # filepath_image_segments = os.path.join(self.base_dir.split(os.sep)[0], f'{imgs_identifier}_image_segments.mmap')
    # filepath_image_patches = os.path.join(self.base_dir.split(os.sep)[0], f'{imgs_identifier}_image_segments_patches.mmap')
    # filepath_image_numbers = os.path.join(self.base_dir.split(os.sep)[0], f'{imgs_identifier}_image_segments_numbers.mmap')
    
    # if all([os.path.exists(filename) for filename in [filepath_image_segments,filepath_image_patches, filepath_image_numbers]]):
    #   return np.memmap(filepath_image_segments, dtype=float, mode='r', shape=self.dims)
    if os.path.exists(filepath):
      print(f"[INFO] loaded image segmentations for {imgs_identifier} from file: {filepath}")
      return np.load(filepath)  
      # return segmentation_dataset['image_segments'], segmentation_dataset['image_numbers'], \
      #   segmentation_dataset['image_patches']

    if param_dict is None:
      param_dict = {}

    for key in ['image_segments', 'image_numbers', 'image_patches']:
      # segmentation_dataset[key] = np.empty((0, self.image_shape[0], self.image_shape[1], 3))
      segmentation_dataset[key] = []

    imgs = self.load_images_from_folder(imgs_identifier, self.num_target_class_imgs)
    if imgs_identifier == self.target_class:
      self.target_class_imgs = imgs
    # image_segments_mmap = np.memmap(
    #   filepath_image_segments, 
    #   dtype=float, 
    #   mode='w+', 
    #   shape=(7250, self.image_shape[0], self.image_shape[1], 3)
    # )
    # image_patches_mmap = np.memmap(
    #   filepath_image_patches, 
    #   dtype=float, 
    #   mode='w+', 
    #   shape=(7250, self.image_shape[0], self.image_shape[1], 3)
    # )
    # image_numbers_mmap = np.memmap(
    #   filepath_image_numbers, 
    #   dtype=int, 
    #   mode='w+', 
    #   shape=(7250,)
    # )
    # current_idx = 0

    for idx, img in tqdm(
      enumerate(imgs), 
      total=len(imgs),
      desc=f'[INFO] Creating segmentations for {imgs_identifier} images'
    ):
      image_superpixels, image_patches = self._return_superpixels(
        img, method, param_dict
      )
      #for superpixel, patch in zip(image_superpixels, image_patches):
      # segmentation_dataset['image_segments'] = np.concatenate(
      #   (segmentation_dataset['image_segments'], image_superpixels), axis=0)
      # segmentation_dataset['image_patches'] = np.concatenate(
      #   (segmentation_dataset['image_patches'], image_patches), axis=0)
      # segmentation_dataset['image_numbers'] = np.concatenate(
      #   (segmentation_dataset['image_numbers'], np.full((image_superpixels.shape[0],), idx)), axis=0)

      segmentation_dataset['image_segments'].extend(image_superpixels)
      segmentation_dataset['image_patches'].extend(image_patches)
      segmentation_dataset['image_numbers'].extend([idx]*len(image_superpixels))
      
    #   image_segments_mmap[current_idx:current_idx+len(image_superpixels)] = np.stack(image_superpixels)
    #   image_patches_mmap[current_idx:current_idx+len(image_superpixels)] = np.stack(image_patches)
    #   image_numbers_mmap[current_idx:current_idx+len(image_superpixels)] = idx
    #   current_idx += len(image_superpixels)
    
    # image_segments_mmap = image_segments_mmap[:current_idx]
    # image_patches_mmap = image_patches_mmap[:current_idx]
    # image_numbers_mmap = image_numbers_mmap[:current_idx]
    
    # create numpy array from lists
    for key in segmentation_dataset:
      segmentation_dataset[key] = np.stack(segmentation_dataset[key])

    if save:
      print(f'Saving image segmentations for {imgs_identifier} in file: {filepath}...') 
      np.savez(filepath, **segmentation_dataset)
    
    return segmentation_dataset
      
  def _return_superpixels(self, img, method='slic', param_dict=None):
    """Returns all patches for one image.

    Given an image, calculates superpixels for each of the parameter lists in
    param_dict and returns a set of unique superpixels by
    removing duplicates. If two patches have Jaccard similarity more than 0.5,
    they are concidered duplicates.

    Args:
      img: The input image
      method: superpixel method, one of slic, watershed, quichsift, or
        felzenszwalb
      param_dict: Contains parameters of the superpixel method used in the form
                of {'param1':[a,b,...], 'param2':[z,y,x,...], ...}. For instance
                {'n_segments':[15,50,80], 'compactness':[10,10,10]} for slic
                method.
    Raises:
      ValueError: if the segementation method is invaled.
    """
    if param_dict is None:
      param_dict = {}
    if method == 'slic':
      n_segmentss = param_dict.pop('n_segments', [15, 50, 80])
      n_params = len(n_segmentss)
      compactnesses = param_dict.pop('compactness', [20] * n_params)
      sigmas = param_dict.pop('sigma', [1.] * n_params)
    elif method == 'watershed':
      markerss = param_dict.pop('marker', [15, 50, 80])
      n_params = len(markerss)
      compactnesses = param_dict.pop('compactness', [0.] * n_params)
    elif method == 'quickshift':
      max_dists = param_dict.pop('max_dist', [20, 15, 10])
      n_params = len(max_dists)
      ratios = param_dict.pop('ratio', [1.0] * n_params)
      kernel_sizes = param_dict.pop('kernel_size', [10] * n_params)
    elif method == 'felzenszwalb':
      scales = param_dict.pop('scale', [1200, 500, 250])
      n_params = len(scales)
      sigmas = param_dict.pop('sigma', [0.8] * n_params)
      min_sizes = param_dict.pop('min_size', [20] * n_params)
    else:
      raise ValueError('Invalid superpixel method!')
    # list to store the segments (at this point still in size
    # of the original image)
    unique_masks = []
    # iterate over the possible settings of the segmentation method
    # (e.g., in case of SLIC, the number of segments in [15,50,80])
    for i in range(n_params):
      # list to store the segments of one specific setting
      param_masks = []
      # create segments
      if method == 'slic':
        segments = segmentation.slic(
            img, n_segments=n_segmentss[i], compactness=compactnesses[i],
            sigma=sigmas[i])
      elif method == 'watershed':
        segments = segmentation.watershed(
            img, markers=markerss[i], compactness=compactnesses[i])
      elif method == 'quickshift':
        segments = segmentation.quickshift(
            img, kernel_size=kernel_sizes[i], max_dist=max_dists[i],
            ratio=ratios[i])
      elif method == 'felzenszwalb':
        segments = segmentation.felzenszwalb(
            img, scale=scales[i], sigma=sigmas[i], min_size=min_sizes[i])
      # iterate over all image segments within the given setting
      for s in range(segments.max()):
        mask = (segments == s).astype(float)
        # only consider segments which contain some informatio
        if np.mean(mask) > 0.001:
          unique = True
          for seen_mask in unique_masks:
            jaccard = np.sum(seen_mask * mask) / np.sum((seen_mask + mask) > 0)
            # only consider mask that do not overlap more than 50 % with existing
            # masks (from other settings)
            if jaccard > 0.5:
              unique = False
              break
          if unique:
            # append segment of image within the current setting 
            # (which passes the two conditions above) to the list
            param_masks.append(mask)
      # append all segments which passed the two conditions in the current
      # setting to the list of all segments
      unique_masks.extend(param_masks)
    superpixels, patches = [], []
    while unique_masks:
      superpixel, patch = self._extract_patch(img, unique_masks.pop())
      superpixels.append(superpixel)
      patches.append(patch)
    return superpixels, patches

  def _extract_patch(self, image, mask):
    """Extracts a patch out of an image.

    Args:
      image: The original image
      mask: The binary mask of the patch area

    Returns:
      image_resized: The resized patch such that its boundaries touches the
        image boundaries
      patch: The original patch. Rest of the image is padded with average value
    """
    mask_expanded = np.expand_dims(mask, -1)
    patch = (mask_expanded * image + (
        1 - mask_expanded) * float(self.average_image_value) / 255)
    ones = np.where(mask == 1)
    h1, h2, w1, w2 = ones[0].min(), ones[0].max(), ones[1].min(), ones[1].max()
    image = Image.fromarray((patch[h1:h2, w1:w2] * 255).astype(np.uint8))
    image_resized = np.array(image.resize(self.image_shape,
                                          Image.BICUBIC)).astype(float) / 255
    return image_resized, patch

  def _load_or_calc_acts(self, model, imgs:np.ndarray, identifier:str, bs:int=50) -> dict:
    """ this function first checks whether the activations for the given layer are 
    already cached (as files) and if so, loads the corresponding file. However,
    if the activations from only one layer are missing, it re-calculates all 
    activations because the intermediate layer activations are obtained via
    forward hooks during one forward pass.

    args:
      model: pre-trained pytorch model instance
      imgs: numpy-array containing the images to compute the activations from
        (should be of shape [num_images, height, width, 3])
      identifier: meaningful identifier for the activations for saving them
      bs: batch size used for calculating the activations

    returns:
      dictionary containing the activations for each layer (if channel_mean=True,
      each value in the dictionary is of shape [num_images, num_channels], otherwise
      each value is of shape [num_images, num_channels, channel_height, channel_width])
    """

    filepath = os.path.join(self.base_dir, 'acts', 'acts_{}_{}.npy'.format(identifier, self.layer))
    if os.path.exists(filepath):
      print(f"[INFO] loaded activations for {identifier} from file: {filepath}")
      return np.load(filepath, allow_pickle=False)
      
    layer_activations = get_layer_activations(model, self.layer, imgs, bs)

    np.save(filepath, layer_activations, allow_pickle=False)
    print(f"[INFO] saved activations for {self.layer} in file: {filepath}")
    return layer_activations
  
  def _cluster(self, acts, method='KM', param_dict=None):
    """Runs unsupervised clustering algorithm on concept activatations.

    Args:
      acts: activation vectors of datapoints points in the bottleneck layer.
        E.g. (number of clusters,) for Kmeans
      method: clustering method. We have:
        'KM': Kmeans Clustering
        'AP': Affinity Propagation
        'SC': Spectral Clustering
        'MS': Mean Shift clustering
        'DB': DBSCAN clustering method
      param_dict: Contains superpixl method's parameters. If an empty dict is
                 given, default parameters are used.

    Returns:
      asg: The cluster assignment label of each data points
      cost: The clustering cost of each data point
      centers: The cluster centers. For methods like Affinity Propagetion
      where they do not return a cluster center or a clustering cost, it
      calculates the medoid as the center  and returns distance to center as
      each data points clustering cost.

    Raises:
      ValueError: if the clustering method is invalid.
    """
    if param_dict is None:
      param_dict = {}
    centers = None
    if method == 'KM':
      n_clusters = param_dict.pop('n_clusters', 25)
      km = cluster.KMeans(n_clusters, n_init='auto', random_state=0)
      d = km.fit(acts)
      centers = km.cluster_centers_
      d = np.linalg.norm(
          np.expand_dims(acts, 1) - np.expand_dims(centers, 0), ord=2, axis=-1)
      asg, cost = np.argmin(d, -1), np.min(d, -1)
    elif method == 'AP':
      damping = param_dict.pop('damping', 0.5)
      ca = cluster.AffinityPropagation(damping)
      ca.fit(acts)
      centers = ca.cluster_centers_
      d = np.linalg.norm(
          np.expand_dims(acts, 1) - np.expand_dims(centers, 0), ord=2, axis=-1)
      asg, cost = np.argmin(d, -1), np.min(d, -1)
    elif method == 'MS':
      ms = cluster.MeanShift(n_jobs=self.num_workers)
      asg = ms.fit_predict(acts)
    elif method == 'SC':
      n_clusters = param_dict.pop('n_clusters', 25)
      sc = cluster.SpectralClustering(
          n_clusters=n_clusters, n_jobs=self.num_workers)
      asg = sc.fit_predict(acts)
    elif method == 'DB':
      eps = param_dict.pop('eps', 0.5)
      min_samples = param_dict.pop('min_samples', 20)
      sc = cluster.DBSCAN(eps, min_samples, n_jobs=self.num_workers)
      asg = sc.fit_predict(acts)
    else:
      raise ValueError('Invalid Clustering Method!')
    if centers is None:  ## If clustering returned cluster centers, use medoids
      centers = np.zeros((asg.max() + 1, acts.shape[1]))
      cost = np.zeros(len(acts))
      for cluster_label in range(asg.max() + 1):
        cluster_idxs = np.where(asg == cluster_label)[0]
        cluster_points = acts[cluster_idxs]
        pw_distances = metrics.euclidean_distances(cluster_points)
        centers[cluster_label] = cluster_points[np.argmin(
            np.sum(pw_distances, -1))]
        cost[cluster_idxs] = np.linalg.norm(
            acts[cluster_idxs] - np.expand_dims(centers[cluster_label], 0),
            ord=2,
            axis=-1)
    return asg, cost, centers

  def discover_concepts(self, model, method='KM', param_dicts=None):
    """Discovers the frequent occurring concepts in the target class.

      Calculates self.dic, a dicationary containing all the informations of the
      discovered concepts in the form of {'bottleneck layer name: bn_dic} where
      bn_dic itself is in the form of {'concepts:list of concepts,
      'concept name': concept_dic} where the concept_dic is in the form of
      {'images': resized patches of concept, 'patches': original patches of the
      concepts, 'image_numbers': image id of each patch}

    Args:
      method: Clustering method.
      activations: If activations are already calculated. If not calculates
                   them. Must be a dictionary in the form of {'bn':array, ...}
      param_dicts: A dictionary in the format of {'bottleneck':param_dict,...}
                   where param_dict contains the clustering method's parametrs
                   in the form of {'param1':value, ...}. For instance for Kmeans
                   {'n_clusters':25}. param_dicts can also be in the format
                   of param_dict where same parameters are used for all
                   bottlenecks.
    """
    if param_dicts is None:
      param_dicts = {}

    # dictionary for storing the image segment activation clusters (=concepts)
    self.concept_dict = {}

    # calculate/load the activations from the target class image segments
    layer_activations = self._load_or_calc_acts(
      model=model,
      imgs=self.segmentation_dataset['image_segments'],
      identifier=f'{self.target_class}_image_segments'
    )

    # if self.channel_mean: average of the channel-dimension (reduces
    # dimensionality for clustering)
    if self.channel_mean:
      layer_acts_flatten = np.mean(layer_activations, axis=(2,3))
    else:
      layer_acts_flatten = np.reshape(layer_activations, [layer_activations.shape[0], -1])

    # cluster target class image segment activations
    self.concept_dict['cluster_label'], self.concept_dict['cluster_cost'], centers = self._cluster(
      layer_acts_flatten, method, param_dicts
    )
    concept_number, self.concept_dict['concepts'] = 0, []

    # iterate over all clusters
    for cluster_idx in tqdm(
      range(self.concept_dict['cluster_label'].max() + 1), 
      desc='[INFO] evaluating target class image segment activations clusters'
      ):
      # returns the indices of all activations belonging to the i-th cluster
      label_idxs = np.where(self.concept_dict['cluster_label'] == cluster_idx)[0]
      # only clusters with more than min_imgs image segments are considered
      if len(label_idxs) > self.min_cluster_size:
        # only the top-max_imgs activations (with minimal cost, i.e., distance to 
        # cluster center) are kept for eac cluster
        concept_costs = self.concept_dict['cluster_cost'][label_idxs]
        concept_idxs = label_idxs[np.argsort(concept_costs)[:self.max_cluster_size]]
        # check which of the images the image segments in the cluster belong to
        # for this all cluster members (not only the top-max_imgs) are considered
        concept_image_numbers = set(self.segmentation_dataset['image_numbers'][label_idxs])

        # Condition based on which concepts are choosen from cluster of image segments

        # check whether the clusters are above average size
        # common_concept = len(label_idxs) > (len(self.concept_dict['cluster_label']) /
        #   self.concept_dict['cluster_label'].max() + 1)
        
        # these conditions check that the image segments present within one cluster belong to
        # more than 50% / 25 % of the total number of target class images
        middle_populated_concept = len(concept_image_numbers) > 0.5 * self.num_target_class_imgs
        # cond1 = len(concept_image_numbers) > 0.75 * self.num_target_class_imgs
        
        # cond2 = middle_populated_concept and common_concept
        
        # if one of the following conditions are fulfilled, the 
        # cluster is considered as concept 
        if middle_populated_concept:
          concept_number += 1
          concept_name = '{}_concept{}'.format(self.target_class, concept_number)
          self.concept_dict['concepts'].append(concept_name)
          self.concept_dict[concept_name] = {
              'images': self.segmentation_dataset['image_segments'][concept_idxs],
              'activations': layer_activations[concept_idxs],
              'patches': self.segmentation_dataset['image_patches'][concept_idxs],
              'image_numbers': self.segmentation_dataset['image_numbers'][concept_idxs]
          }
          self.concept_dict[concept_name + '_center'] = centers[cluster_idx]
          tqdm.write(
            f"[INFO] cluster {cluster_idx}/{self.concept_dict['cluster_label'].max() + 1} "
            f"of size {len(label_idxs)}/{len(self.concept_dict['cluster_label'])} passed conditions "
            f"({middle_populated_concept}) and is considered a concept"
          )

    self.concept_dict.pop('cluster_label', None)
    self.concept_dict.pop('cluster_cost', None)

  def save_concepts(self):
    """
    Saves discovered concept's images and patches
    """
    
    concept_dir = os.path.join(self.base_dir, 'concepts')
    if os.path.exists(concept_dir):
      shutil.rmtree(concept_dir)
      print(f"[INFO] Deleted existing concepts in folder: {concept_dir}")
    os.makedirs(concept_dir)
    
    # iterate over all concepts
    for concept in tqdm(self.concept_dict['concepts'], desc='[INFO] saving concept images/patches'):
      # set directory names
      patches_dir = os.path.join(self.base_dir, 'concepts', concept + '_patches')
      images_dir = os.path.join(self.base_dir, 'concepts', concept)
      
      os.makedirs(patches_dir)
      os.makedirs(images_dir)
      
      # create RGB images with image values between 0-256
      patches = (np.clip(self.concept_dict[concept]['patches'], 0, 1) * 256).astype(np.uint8)
      images = (np.clip(self.concept_dict[concept]['images'], 0, 1) * 256).astype(np.uint8)
      
      image_numbers = self.concept_dict[concept]['image_numbers']
      image_addresses, patch_addresses = [], []
      
      # save the images in format "iterator_imagenumber"
      for i in range(len(images)):
        image_name = '0' * int(np.ceil(2 - np.log10(i + 1))) + '{}_{}'.format(
          i + 1, image_numbers[i]
        )
        patch_addresses.append(os.path.join(patches_dir, image_name + '.png'))
        image_addresses.append(os.path.join(images_dir, image_name + '.png'))
      save_images(patch_addresses, patches)
      save_images(image_addresses, images)

  # def _train_cavs(self, concept_name, concept_acts, all_random_acts):
    
  #   # initialize dictionary to store the CAVs 
  #   # for each of the random datasets
  #   cav = {}

  #   for id, random_acts in all_random_acts.items():
      
  #     concept_dataset = ConceptDataset(
  #       data=torch.tensor(np.concatenate(
  #         [concept_acts.reshape(concept_acts.shape[0], -1), random_acts.reshape(random_acts.shape[0], -1)], axis=0
  #         )),
  #       labels=torch.cat([torch.ones(concept_acts.shape[0]), torch.full((random_acts.shape[0],), -1)], dim=0)
  #     )
  #     # svm_model = LinearSVM(concept_acts.shape[-1])
  #     svm_model = torch.nn.Linear(concept_dataset.data.size(-1), 1, bias=False)

  #     # Define the loss function (hinge loss for SVM)
  #     # criterion = torch.nn.MarginRankingLoss(margin=1.0)

  #     # Define the optimizer (e.g., Stochastic Gradient Descent)
  #     optimizer = torch.optim.SGD(svm_model.parameters(), lr=0.01)

  #     dataloader = torch.utils.data.DataLoader(concept_dataset, batch_size=50, shuffle=True)

  #     for epoch in range(10): # FIXME: Make it a parameter
  #       total_correct = 0
  #       total_samples = 0
  #       sum_loss = 0
  #       for b_data, b_labels in dataloader:
  #           optimizer.zero_grad()
  #           b_output = svm_model(b_data).squeeze()
  #           weight = svm_model.weight.squeeze()

  #           loss = torch.mean(torch.clamp(1 - b_labels * b_output, min=0))
  #           loss += 0.01 * (weight.t() @ weight) / 2.0 # FIXME: Make 0.01 a parameter
  #           # loss = criterion(b_output, b_labels.view(-1, 1))
  #           loss.backward()
  #           optimizer.step()

  #           sum_loss += float(loss)
  #           total_correct += (b_output[b_labels == -1] < 0).sum() + (b_output[b_labels == 1] > 0).sum()
  #           total_samples += b_labels.size(0)

  #       accuracy = total_correct / total_samples

  #       # print(f'Epoch [{epoch + 1}/10], Loss: {sum_loss:.4f}, Accuracy: {accuracy:.4f}')
      
  #     print(f'[{concept_name}|{id}] Accuracy: {accuracy:.4f}', end='\r')

  #     weight_vector = svm_model.weight.data.cpu().numpy()

  #     cav[id] = {
  #       'weight_vector': weight_vector / np.linalg.norm(weight_vector, axis=1)[:, None],
  #       'accuracy': accuracy
  #     }
    
  #   return cav

  def _train_cavs_sklearn(self, concept_acts, all_random_acts) -> dict:
    
    # initialize dictionary to store the CAVs 
    # for each of the random datasets
    cav = {}

    # to make sure positive and negative examples are balanced, truncate
    # activations to concept size if it is smaller than self.max_cluster_size
    min_num_acts = np.min([concept_acts.shape[0], self.max_cluster_size])
      
    for random_id, random_acts in all_random_acts.items():
      # intialize SVM
      lm = linear_model.SGDClassifier(alpha=0.01)
      x = np.concatenate(
        [
          concept_acts.reshape(concept_acts.shape[0], -1)[:min_num_acts], 
          random_acts.reshape(random_acts.shape[0], -1)[:min_num_acts]
        ],
        axis=0
      )
      y = np.concatenate(
        [np.ones(min_num_acts), np.zeros(min_num_acts)],
        axis=0
      )
      X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
      lm.fit(X_train, y_train)
      y_pred = lm.predict(X_test)

      cav[random_id] = {
        'weight_vector': np.expand_dims(lm.coef_[0], 0) / np.linalg.norm(lm.coef_[0], ord=2),
        'accuracy': metrics.accuracy_score(y_pred, y_test)
      }
    
    return cav
  
  def calculate_cavs(self, model, min_acc=0.):
    """Calculates cavs for all discovered concepts.

    This method calculates and saves CAVs for all the discovered concepts
    versus all random concepts in all the bottleneck layers

    Args:
      min_acc: Delete discovered concept if the average classification accuracy
        of the CAV is less than min_acc
      ow: If True, overwrites an already calcualted cav.

    Returns:
      A dicationary of classification accuracy of linear boundaries orthogonal
      to cav vectors
    """

    # dictionary to store the CAV for each concepts 
    # for each of the random datasets
    self.cavs = {}

    # if os.path.exists(filepath):
    #   # Save the dictionary as a pickle file  
    #   with open(filepath, 'rb') as file:
    #       self.cavs = pickle.load(file)
    #       print(f"[INFO] loaded CAVs from file: {filepath}")
    #   return

    # calculate the activations for the datasets containing random images 
    # (are used as negative examples to train the CAVs)
    all_random_acts = {}
    # np.random.seed(42)
    
    random_segments = self.create_patches(
      imgs_identifier='random',
      param_dict={'n_segments': [15,50,80]},
      save=True
    )['image_segments']
    
    for random_dataset_folder in ['random_{}'.format(i) for i in np.arange(self.num_random_datasets)]:
      
      selected_indices = np.random.choice(
        random_segments.shape[0], 
        self.max_cluster_size, # select as max num of examples of concepts
        replace=False # same index cannot occur twice
      )
      
      # calculate activations for random image segments
      all_random_acts[random_dataset_folder] = self._load_or_calc_acts(
        model=model,
        imgs=random_segments[selected_indices],
        # imgs=self.load_images_from_folder(random_dataset_folder, self.max_cluster_size),
        identifier=random_dataset_folder,
      )
    
    # train the CAVs for each concept vs. all random counterparts
    for concept_name in tqdm(self.concept_dict['concepts'], desc='[INFO] learning CAVs'):
      concept_acts = self.concept_dict[concept_name]['activations']

      self.cavs[concept_name] = self._train_cavs_sklearn(concept_acts, all_random_acts)

      # if the accuracy of the trained linear classifier is below a threshold,
      # delete the corresponding concept
      if np.mean([cav['accuracy'] for cav in self.cavs[concept_name].values()]) < min_acc:
        self._delete_concept(concept_name)

    # train the CAV for a random concept vs. all random counterparts (later used
    # for statistical testing)
    
    selected_indices = np.random.choice(
      random_segments.shape[0], 
      self.max_cluster_size, # select as max num of examples of concepts
      replace=False # same index cannot occur twice
    )
    self.cavs['random'] = self._train_cavs_sklearn(
      concept_acts=self._load_or_calc_acts(
        model=model,
        imgs=random_segments[selected_indices],
        # imgs=self.load_images_from_folder('random', self.max_cluster_size),
        identifier='random',
      ), 
      all_random_acts=all_random_acts,
    )
    
    # save the CAVs as pickle file
    filepath = os.path.join(self.base_dir, f'cavs_{self.layer}.pkl')
    with open(filepath, 'wb') as file:
      pickle.dump(self.cavs, file)
      print(f"[INFO] saved CAVs in file {filepath}")

    del random_segments
    
  def _sort_concepts(self):
    # define a function to calculate the mean value of a list
    def mean_value(values):
      return sum(values) / len(values) if len(values) > 0 else 0
    # sort the list of concepts based on the mean value of the TCAV scores
    sorted_concepts = sorted(self.concept_dict['concepts'], key=lambda concept: mean_value(self.tcav_scores[concept]), reverse=True)
    # replace the list of concept in self.concept_dict with the sorted version
    self.concept_dict['concepts'] = sorted_concepts

  def calc_tcavs(self, model, cls_id:int=None, test:bool=False, sort:bool=True):
    """calculates TCAV scores of all the discovered concepts for
    the target class using all the calculated CAVs. It later sorts concepts
    based on their TCAV scores.

    Args:
      test: If true, perform statistical testing and removes concepts that don't
        pass
      sort: If true, it will sort concepts in each bottleneck layers based on
        average TCAV score of the concept.
      tcav_score_images: Target class images used for calculating tcav scores.
        If None, the target class source directory images are used.

    Returns:
      A dictionary of the form {'bottleneck layer':{'concept name':
      [list of tcav scores], ...}, ...} containing TCAV scores.
    """

    # dict to store the TCAV scores for each discovered concept
    self.tcav_scores = {}

    # this is the case when the target class image segments 
    # were loaded from cache
    if self.target_class_imgs is None:
      self.target_class_imgs = self.load_images_from_folder(
        folder_name=self.target_class, 
        max_imgs=self.num_target_class_imgs*2
      )
      
    # take different target class images than those used to create the 
    # image segments (and therewith the concepts)
    target_class_imgs = self.target_class_imgs[-self.num_target_class_imgs:]

    # get the gradients for the target class images of the specified layer
    gradients = get_layer_activations(
      model=model, 
      layer= self.layer, 
      imgs=target_class_imgs, 
      get_grads_for_cls_id=cls_id
    )
    
    # iterate over all concepts (+ a concept trained on random images which
    # is used to later perform statistical testing)
    for concept_name in self.concept_dict['concepts'] + ['random']:
      # iterate over the CAV trained with different random counterparts
      self.tcav_scores[concept_name] = []
      for cav in self.cavs[concept_name].values():
        # take the inner product of the gradient vector of images from the 
        # target class with the CAV vector
        prod = np.sum(
          gradients.reshape(gradients.shape[0], -1) * cav['weight_vector'], 
          axis=-1
        )
        # calculate the share of target class images for which the TCAV value
        # is above 0 (i am not 100 % sure why they use "<" but I assume that
        # this compensates for the direction of the CAV vectors)
        self.tcav_scores[concept_name].append(np.mean(prod < 0))
        
    if test:
      # perform statistical testing of TCAV scores for each concept and remove
      # concepts which do not pass the test
      self.test_and_remove_concepts()
    if sort:
      # sort conecepts based on their average TCAV score (=importance)
      self._sort_concepts()

  def _do_statistical_testings(self, i_ups_concept, i_ups_random):
    """Conducts ttest to compare two set of samples.

    In particular, if the means of the two samples are staistically different.

    Args:
      i_ups_concept: samples of TCAV scores for concept vs. randoms
      i_ups_random: samples of TCAV scores for random vs. randoms

    Returns:
      p value
    """
    min_len = min(len(i_ups_concept), len(i_ups_random))
    _, p = stats.ttest_rel(i_ups_concept[:min_len], i_ups_random[:min_len])
    return p

  def test_and_remove_concepts(self, p_threshold:int=0.01):
    """using TCAV socres of the discovered concepts vs. the a concept trained with
    random images, this function performs statistical testing and removes concepts 
    that do not pass the test

    Args:
        p_threshold (int, optional): threshold on p-value to decide whether to
          delete a given concept. Defaults to 0.01.
    """
    
    # for each discovered concept
    for concept_name in self.concept_dict['concepts']:
      # calculate the p-value given the TCAV score of the concept 
      # vs. a concept trained with random images
      pvalue = self._do_statistical_testings(
        self.tcav_scores[concept_name], 
        self.tcav_scores['random']
      )
      # if the p-value succeeds a given threshold
      if pvalue > p_threshold:
        # delete the concept from self.concept_dict
        self._delete_concept(concept_name)

  def _delete_concept(self, concept):
    """Removes a discovered concepts if it's not already removed.

    Args:
      bn: Bottleneck layer where the concepts is discovered.
      concept: concept name
    """
    self.concept_dict.pop(concept, None)
    if concept in self.concept_dict['concepts']:
      self.concept_dict['concepts'].pop(self.concept_dict['concepts'].index(concept))
    
  def save_tcav_report(self):
    """for each concept, save the CAV accuracies, the mean of the TCAV values
    and the corresponding p-values in a CSV-file"""
    
    # store concept-identifiers in a list
    concept_names = self.concept_dict['concepts']
    # store mean of corresponding concept accuracies in a list
    cav_accuracies = [
      np.mean(
        [cav['accuracy'] for cav in self.cavs[concept_name].values()]
      ) for concept_name in concept_names
    ]
    # store mean of corresponding TCAV scores in a list 
    tcav_means = [
      np.mean(self.tcav_scores[concept_name]) for concept_name in concept_names
    ]
    # store corresponfing p-value in a list 
    p_values = [
      self._do_statistical_testings(
        self.tcav_scores[concept_name], self.tcav_scores['random']
      ) for concept_name in concept_names
    ]

    # save results in CSV file
    filepath = os.path.join(self.base_dir, 'tcav_results.csv')
    with open(filepath, 'w', newline='') as csv_file:
      writer = csv.writer(csv_file)
      writer.writerow(['Concept', 'CAV Accuracy', 'TCAV Mean', 'P-Value'])
      for concept_name, cav_accuracy, tcav_mean, p_value in zip(concept_names, cav_accuracies, tcav_means, p_values):
          writer.writerow([concept_name, cav_accuracy, tcav_mean, p_value])

    print(f"[INFO] saved results in csv file: {filepath}") 
  
  def plot_concepts(self, num:int=10, mode='diverse', concepts=None):
    """Plots examples of discovered concepts.

    Args:
      num: Number of images to print out of each concept
      mode: If 'diverse', it prints one example of each of the target class images
        is coming from. If 'radnom', randomly samples exmples of the concept. If
        'max', prints out the most activating examples of that concept.
      concepts: If None, prints out examples of all discovered concepts.
        Otherwise, it should be either a list of concepts to print out examples of
        or just one concept's name

    Raises:
      ValueError: If the mode is invalid.
    """
    if concepts is None:
      concepts = self.concept_dict['concepts']
    elif not isinstance(concepts, list) and not isinstance(concepts, tuple):
      concepts = [concepts]
    num_concepts = len(concepts)
    plt.rcParams['figure.figsize'] = num * 2.1, 4.3 * num_concepts
    fig = plt.figure(figsize=(num * 2, 4 * num_concepts))
    outer = gridspec.GridSpec(num_concepts, 1, wspace=0., hspace=0.3)
    for n, concept in enumerate(concepts):
      inner = gridspec.GridSpecFromSubplotSpec(
          2, num, subplot_spec=outer[n], wspace=0, hspace=0.1)
      concept_images = self.concept_dict[concept]['images']
      concept_patches = self.concept_dict[concept]['patches']
      concept_image_numbers = self.concept_dict[concept]['image_numbers']
      if mode == 'max':
        idxs = np.arange(len(concept_images))
      elif mode == 'random':
        idxs = np.random.permutation(np.arange(len(concept_images)))
      elif mode == 'diverse':
        idxs = []
        while True:
          seen = set()
          for idx in range(len(concept_images)):
            if concept_image_numbers[idx] not in seen and idx not in idxs:
              seen.add(concept_image_numbers[idx])
              idxs.append(idx)
          if len(idxs) == len(concept_images):
            break
      else:
        raise ValueError('Invalid mode!')
      idxs = idxs[:num]
      for i, idx in enumerate(idxs):
        ax = plt.Subplot(fig, inner[i])
        ax.imshow(concept_images[idx])
        ax.set_xticks([])
        ax.set_yticks([])
        if i == int(num / 2):
          ax.set_title(concept)
        ax.grid(False)
        fig.add_subplot(ax)
        ax = plt.Subplot(fig, inner[i + num])
        mask = 1 - (np.mean(concept_patches[idx] == float(
            self.average_image_value) / 255, -1) == 1)
        image = self.target_class_imgs[concept_image_numbers[idx]]
        ax.imshow(segmentation.mark_boundaries(image, mask, color=(1, 1, 0), mode='thick'))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(str(concept_image_numbers[idx]))
        ax.grid(False)
        fig.add_subplot(ax)
      
      filepath = os.path.join(self.base_dir, f'{self.layer}_concept_examples.png')
      fig.savefig(filepath)
      
    print(f"[INFO] saved example images for each concept in file: {filepath}") 
      
    plt.clf()
    plt.close(fig)