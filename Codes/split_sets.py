import random
import time
from collections import defaultdict

GROUP_SIZE = 3
TEST_RATIO = 0.2

def get_group_starts(num_files, group_size):
    return list(range(0, num_files, group_size))

def get_group_starts(num_files, group_size):
    return list(range(0, num_files, group_size))

def split_folds(n, num_folds=5, actor_audio_no=246, group_size = GROUP_SIZE):
    random.seed(time.time())

    actor_starts = get_group_starts(n, actor_audio_no)
    group_starts = get_group_starts(n, group_size)

    actor_no = len(actor_starts)

    actor_per_fold = actor_no // num_folds

    folds = [[i for i in range(actor_starts[fold * actor_per_fold], actor_audio_no * (fold + 1) * actor_per_fold)] for fold in range(num_folds)]
    original_folds = [group_starts[f * (actor_per_fold * actor_audio_no // group_size) : (f + 1) * (actor_per_fold * actor_audio_no // group_size)] for f in range(num_folds)]
    
    return folds, original_folds