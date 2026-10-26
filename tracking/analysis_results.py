import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist

trackers = []
# dataset_name = 'dtb70'
# dataset_name = 'uavdt'
dataset_name = 'visdrone2018'
# dataset_name = 'uav123'
# dataset_name = 'uav123_10fps'
# dataset_name = 'uavseaship'
# dataset_name = 'uavtrack112'
# dataset_name = 'uavtrack112l'
# dataset_name = 'lasot'
# dataset_name = 'otb'
# dataset_name = 'nfs'

"""dmftrack"""

# trackers.extend(trackerlist(name='dmftrack_st2', parameter_name='deit_tiny_patch16_224', dataset_name=dataset_name,
#                             run_ids=None, display_name='DMFTrack(deit)')) # add for test DMFTrack
trackers.extend(trackerlist(name='dmftrack_st2', parameter_name='vit_tiny_patch16_224', dataset_name=dataset_name,
                            run_ids=None, display_name='DMFTrack(vit)')) # add for test DMFTrack
# trackers.extend(trackerlist(name='dmftrack_st2', parameter_name='eva02_tiny_patch14_224', dataset_name=dataset_name,
#                             run_ids=None, display_name='DMFTrack(eva02)')) # add for test DMFTrack


dataset = get_dataset(dataset_name)

print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'norm_prec', 'prec'))

