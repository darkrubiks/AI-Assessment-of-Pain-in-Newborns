import sys
from XAI.concept_discovery import *
import argparse
import json
import torch

from models import *


def main(args):
    
    # create directories to load/save cached values and results
    base_dir = create_dirs(
        args.working_dir,
        args.target_class, 
        args.model_name, 
        args.layer
    )
    
    # load PyTorch model
    #model = NCNN()
    model = VGGFace()
    model.load_state_dict(torch.load(r'experiments\VGGFace_FINAL\20250826_1353_VGGFace\Model\best_model.pt'))
    
    # load ImageNet class label file and extract 
    # the index for args.target_class
    class_id = None

    # Creating the ConceptDiscovery class instance
    cd = ConceptDiscovery(
        target_class=args.target_class,
        layer=args.layer,
        source_dir=args.source_dir,
        base_dir=base_dir,
        num_random_datasets=args.num_random_datasets,
        channel_mean=True,
        input_shape=(224,224),
        min_cluster_size=args.min_cluster_size,
        max_cluster_size=args.max_cluster_size,
        num_target_class_imgs=args.num_target_class_imgs,
        num_workers=args.num_parallel_workers
    )
    
    # These are all image segments which are created from the images in $target_class folder
    cd.segmentation_dataset = cd.create_patches(
        imgs_identifier=args.target_class, 
        param_dict={'n_segments': [15, 50, 80]},
        save=True
    )

    # Using the activations for all image segments for each layer, create clusters and 
    # only keep those clusters, which fulfill certain conditions (see code or paper)
    cd.discover_concepts(model=model, method='KM', param_dicts={'n_clusters': 25})

    # free memory by deleting variables not needed anymore
    del cd.segmentation_dataset

    # For each cluster (= concept), save the corresponding image segments 
    # and the corresponding patch within the original image
    cd.save_concepts()

    # for each activations cluster, calculate the Concept Activcation Vectors (CAV)
    cd.calculate_cavs(model)
    
    cd.calc_tcavs(model, class_id, test=False)
    
    # save CAV test accuracies and TCAV scores + p-values in text-file
    cd.save_tcav_report()
    
    # # Plot examples of discovered concepts
    cd.plot_concepts(num=10)

def parse_arguments(argv):
    """Parses the arguments passed to the run.py script."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--source_dir', 
        type=str,
        help='directory where the class- and random image datasets are saved', 
        default=r'D:\### DOUTORADO\Mestrado\RGU\ACE\source'
    )
    parser.add_argument(
        '--working_dir', 
        type=str,
        help='directory where the cached values and results are saved/loaded', 
        default=r'D:\### DOUTORADO\Mestrado\RGU\ACE'
    )

    parser.add_argument(
        '--model_name', 
        type=str,
        help='model name', 
        default='VGGFace'
    )

    parser.add_argument(
        '--target_class', 
        type=str,
        help='name of the target class to be interpreted', 
        default='pain'
    )
    parser.add_argument(
        '--layer', 
        type=str,
        help='name of the layer from which the activations should be calculated',
        default='VGGFace'
    )
    parser.add_argument(
        '--num_random_datasets', 
        type=int,
        help="Number of random datasets used to calculate CAVs",
        default=20
    )
    parser.add_argument(
        '--num_target_class_imgs', 
        type=int,
        help="number of images of target class used to create image segments",
        default=50
    )
    parser.add_argument(
        '--min_cluster_size', 
        type=int,
        help="minimum number of image segments of each cluster",
        default=30
    )
    parser.add_argument(
        '--max_cluster_size', 
        type=int,
        help="Maximum number of image segments (with minimal cost) considered for each cluster",
        default=50
    )
    parser.add_argument(
        '--num_parallel_workers', 
        type=int,
        help="Number of parallel jobs.",
        default=0
    )
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

