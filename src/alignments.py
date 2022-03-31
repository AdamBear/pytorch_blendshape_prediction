from decimal import Decimal
import math

# loads alignments from CTM file and tiles according to framerate
def preprocess_alignments(path, phone_ids=None, framerate=59.97, remap_silence=("sil", [ "spn"])):
    frame_len_secs = 1 / framerate
    i = 0
    lines = [ line.strip().split(' ') for line in open(path, 'r') ]
    frames = []
    for line in lines:
        start = float(line[2])
        duration = float(line[3])
        end = start + duration
        label = line[4]
        start_frame = math.floor(start / frame_len_secs) # round down for start frames
        end_frame = math.ceil(end / frame_len_secs) # round up for end frames
        
        if len(frames) < end_frame:
            frames += [ None ] * (end_frame - len(frames))
        label = label.lower()
        if label in remap_silence[1]:
            label = remap_silence[0]
        frames[start_frame:end_frame] =  [ phone_ids[label ] ] * (end_frame - start_frame)
    assert(len(frames) == math.ceil(end / frame_len_secs))
    return frames