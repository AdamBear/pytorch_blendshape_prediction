from decimal import Decimal
import math
import itertools

# expands/reduces Kaldi frame-based alignments to a target framerate
# assumes Kaldi alignment frames are 10ms long
# also we take the opportunity to replace "SPN" with "SIL" to maintain consistency
def preprocess_alignments(alignments, phone_ids=None, framerate=59.97, remap_silence=("sil", [ "spn"])):
    src_frame_len_in_secs = 0.01
    target_frame_len_in_secs = 1 / framerate

    src_length_in_frames = sum([x[1] for x in alignments])
    src_length_in_secs = src_length_in_frames * src_frame_len_in_secs
    target_length_in_frames = framerate * src_length_in_secs

    conversion_factor = src_frame_len_in_secs / target_frame_len_in_secs
    # print(f"conversion_factor {conversion_factor} src_length_in_frames {src_length_in_frames} src_length_in_secs {src_length_in_secs} target_frame_len_in_secs {target_frame_len_in_secs} target_length_in_frames {target_length_in_frames}")
    frames = []
    phone_index = 0
    src_frame_offset = 0
    if conversion_factor < 1:   
        for target_frame_num in range(int(target_length_in_frames)):
            target_frame_end = target_frame_num * target_frame_len_in_secs
            
            while phone_index < len(alignments) and target_frame_end > (alignments[phone_index][1] + src_frame_offset) * src_frame_len_in_secs:
                src_frame_offset += alignments[phone_index][1]
                phone_index += 1
            if phone_index >= len(alignments):
                break
            
            phone_sym = alignments[phone_index][0]
            phone_sym = remap_silence[0] if phone_sym.lower() in remap_silence[1] else phone_sym
            phone_id = phone_ids[phone_sym]
            frames += [phone_id]
    return frames