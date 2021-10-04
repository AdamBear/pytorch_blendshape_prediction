indir=$1
outdir=$2
if [ $# -lt 2 ] ; then
	echo -e "Usage: ./segment.sh <indir> <outdir>\nFinds every .wav file in the provided <indir>, looks for a file named audio_alignments.txt in the same directory,\nnormalizes the viseme CSV according to the alignments file and copies the audio, transcript and viseme file to <outdir>" && exit -1;
fi

OLDIFS=$IFS
IFS=$'\n'

alignment_files=$(find $(realpath $indir) -name "audio_alignments.txt")
echo "$alignment_files"

# in every directory, we look for a single file called audio_alignments.txt
# this contains the start/stop times and accompanying transcript for an audio recording
for alignment_file in $alignment_files; do
    current=$(dirname $alignment_file)
   
    # TODO - what if multiple source files (e.g. recorded via multiple microphones)?
    # pitch-shift the first audio file in the directory
    audio_file="${current}/$(basename $current).wav"

    for ratio in 0.7 0.8 0.9 1.1 1.2 1.3; do
        ffmpeg -n -i $audio_file -af "asetrate=44100*$ratio, aresample=44100, atempo=1/$ratio" $(echo "$audio_file" | sed "s/\.wav/$ratio.wav/g")
    done

    # now find all audio files
    audio_files=$(find $current -name "*.wav")

    if [ -z "$audio_files" ]; then
    	echo "No audio file matching $alignment_file" && exit -1;
    fi
    
    # make sure we use the calibrated blendshapes
    blendshapes=$(find $current -name "*_cal.csv")

    audio_alignments=$(cat $alignment_file | awk --field-separator=$'\t' '{printf "%s\t%f\t%s\n",$1,($2-$1),$3}' | sed -E "/^[[:space:]]$/d");
    echo "Generating blendshape CSV for $alignment_file";
    header=$(head -n1 $blendshapes)

    i=0; 

    # each line in the alignments file corresponds to a single phrase
    # for each line, extract the start/end time, find the matching period in the CSV and output to a new CSV
    for line in $audio_alignments; do
        if [ -z "$line" ]; then
            continue;
        fi 
        start=$(echo $line | cut -f1); 
        duration=$(echo $line | cut -f2); 

        if [ -z "$start" ] || [ -z "$duration" ]; then
            echo "Couldn't find start or duration for $alignment_file" && exit;
        fi

        transcript=$(echo $line | cut -f3);
        len=$(echo "$transcript" | wc -m)
        if [ $len -lt 2 ]; then
            echo "Error processing $alignment_file";
            exit -1;
        fi
        bs_data=$(tail -n+2 $blendshapes | awk -F'[:,]' -v start=$start -v duration=$duration -F':' '{
            hour=$1; min=$2;
            second=($3+($4/59.97));
            time=(hour*60*60)+((min*60)+second);

            if(FNR==1) { 
                offset=time 
            }; 
            normalized=(time-offset); 
            if (normalized > start && normalized < (start+duration)) { 
                print ; 
            }
        }')
        for audio_file in $audio_files; do
            audio_id=$(basename $audio_file | sed "s/\.wav//g")
            echo "$transcript" > $outdir/"${audio_id}_$i".txt
            ffmpeg -n -i $audio_file -ss $start -t $duration -ar 22050 "$outdir/${audio_id}_$i.wav" 2>/dev/null 
            echo "$header" > "$outdir/${audio_id}_$i.csv"
            echo "$bs_data" >> "$outdir/${audio_id}_$i.csv"; 
            if [ $(wc -l "$outdir/${audio_id}_$i.csv" | cut -d' ' -f1) -lt 2 ]; then
                echo "Error handling blendshapes from $blendshapes";
            fi
        done

        i=$((i+1)); 
    done
done
IFS=$OLDIFS

