import pandas as pd

def preprocess_viseme(path,
                      blendshapes=None,
                      framerate=30):

    csv = pd.read_csv(path)
    
    factor = 1 / framerate 

    frames = []
    
    start = None
    
    for i, row in csv.iterrows():

      hour, minute, second, frame = row["Timecode"].split(":")
      hour, minute, second, frame = float(hour), float(minute), float(second), float(frame)
      second_offset = (hour * 60 * 60) + (minute * 60) + second + (frame / 59.97)
      if start is None:
        start = second_offset
      second_offset -= start
      if second_offset >= len(frames) * factor:
        frames += [ row[blendshapes if blendshapes is not None else [c for c in csv.columns if c not in ["Timecode", "BlendShapeCount"]]].values ]

    return pd.DataFrame(frames)


LIVE_LINK_FACE_HEADER = "Timecode,BlendShapeCount,eyeBlinkRight,eyeLookDownRight,eyeLookInRight,eyeLookOutRight,eyeLookUpRight,eyeSquintRight,eyeWideRight,eyeBlinkLeft,eyeLookDownLeft,eyeLookInLeft,eyeLookOutLeft,eyeLookUpLeft,eyeSquintLeft,eyeWideLeft,jawForward,jawRight,jawLeft,jawOpen,mouthClose,mouthFunnel,mouthPucker,mouthRight,mouthLeft,mouthSmileRight,mouthSmileLeft,mouthFrownRight,mouthFrownLeft,mouthDimpleRight,mouthDimpleLeft,mouthStretchRight,mouthStretchLeft,mouthRollLower,mouthRollUpper,mouthShrugLower,mouthShrugUpper,mouthPressRight,mouthPressLeft,mouthLowerDownRight,mouthLowerDownLeft,mouthUpperUpRight,mouthUpperUpLeft,browDownRight,browDownLeft,browInnerUp,browOuterUpRight,browOuterUpLeft,cheekPuff,cheekSquintRight,cheekSquintLeft,noseSneerRight,noseSneerLeft,tongueOut,HeadYaw,HeadPitch,HeadRoll,LeftEyeYaw,LeftEyePitch,LeftEyeRoll,RightEyeYaw,RightEyePitch,RightEyeRoll".split(',')

BLENDER_HEADER = "Timecode,BlendShapeCount,EyeBlinkLeft,EyeLookDownLeft,EyeLookInLeft,EyeLookOutLeft,EyeLookUpLeft,EyeSquintLeft,EyeWideLeft,EyeBlinkRight,EyeLookDownRight,EyeLookInRight,EyeLookOutRight,EyeLookUpRight,EyeSquintRight,EyeWideRight,JawForward,JawRight,JawLeft,JawOpen,MouthClose,MouthFunnel,MouthPucker,MouthRight,MouthLeft,MouthSmileLeft,MouthSmileRight,MouthFrownLeft,MouthFrownRight,MouthDimpleLeft,MouthDimpleRight,MouthStretchLeft,MouthStretchRight,MouthRollLower,MouthRollUpper,MouthShrugLower,MouthShrugUpper,MouthPressLeft,MouthPressRight,MouthLowerDownLeft,MouthLowerDownRight,MouthUpperUpLeft,MouthUpperUpRight,BrowDownLeft,BrowDownRight,BrowInnerUp,BrowOuterUpLeft,BrowOuterUpRight,CheekPuff,CheekSquintLeft,CheekSquintRight,NoseSneerLeft,NoseSneerRight,TongueOut,HeadYaw,HeadPitch,HeadRoll,LeftEyeYaw,LeftEyePitch,LeftEyeRoll,RightEyeYaw,RightEyePitch,RightEyeRoll"

BLENDER_TO_LL_MAP = {h:(h[0].lower() + h[1:]) if h not in ["Timecode","BlendShapeCount","HeadYaw","HeadPitch","HeadRoll","LeftEyeYaw","LeftEyePitch","LeftEyeRoll","RightEyeYaw","RightEyePitch","RightEyeRoll"]  else h for h in BLENDER_HEADER.split(",") }

def convert_livelink_to_blender(csv_df, config):
  return csv_df.rename(columns=BLENDER_TO_LL_MAP)
  df = preprocess_viseme("data/training/speaker_3/MySlate_22_Nic_ps-15_20.csv", pad_len_in_secs=config["paddedAudioLength"], 
                                   target_framerate=config["frameRate"])
  df = new_to_old(df)
  df = df[df["Timecode"] != 0]
  df.to_csv("output/predicted.csv", index=False)

  output_indices = [header.index(x[0].lower() + x[1:]) for x in config.sourceKeys]

  # # the prediction in LiveLink Face format
  # with open("output/prediction.csv", "w") as outfile:
  #     outfile.write(",".join(header) + "\n")
  #     timer_ms = 0
  #     for t in range(preds.shape[1]):
  #         output = [str(0)] * len(header)
  #         second = str(int(timer_ms // 1000)).zfill(2)
  #         frame = (timer_ms % 1000) * model_config["frameRate"] / 1000
  #         output[0] = f"00:00:{second}:{frame}"
  #         for viseme in range(len(output_viseme_indices)): 
  #             output[output_indices] = str(export_y[viseme,t].item())
  #         timer_ms += (1 / model_config["frameRate"]) * 1000
  #         outfile.write(",".join(output) + "\n")
          
# write the prediction in App format
def write_preds(preds, target_names, outpath="output/prediction_app.csv"):
  with open(outpath, "w") as outfile:
      outfile.write(",".join(target_names) + "\n")
      for step in range(preds.size(1)):
          for weight in range(preds.size(2)):
              outfile.write(str(preds[0,step,weight].item()))
              outfile.write(",")
          outfile.write("\n")