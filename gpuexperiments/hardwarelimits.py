# data comes from http://developer.download.nvidia.com/compute/cuda/CUDA_Occupancy_calculator.xls
# and from http://www.hardware.fr/articles/948-2/gp104-7-2-milliards-transistors-16-nm.html "La mémoire partagée des SM
# du GP100 passe de 96 à 64 Ko mais elle n'est associée qu'à deux partitions au lieu de 4 ce qui indique en réalité une
# augmentation relative de 33%."
compute_capability_characteristics = {
    '5.0': {'shared_memory_per_sm': 65536},
    '5.2': {'shared_memory_per_sm': 98304},
    '6.1': {'shared_memory_per_sm': 65536}
}
