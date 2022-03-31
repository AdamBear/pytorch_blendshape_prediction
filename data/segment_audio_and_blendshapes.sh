indir=$1
outdir=$2
if [ $# -lt 2 ] ; then
	echo "Usage: ./segment_audio_and_blendshapes.sh indir outdir [ --use-raw ] [ --nostretch ]
Finds every .wav file in the provided <indir>, looks for a file named audio_alignments.txt in the same directory,\nnormalizes the viseme CSV according to the alignments file and copies the audio, transcript and viseme file to <outdir>" && exit -1;
fi

shift
shift
use_raw=0
nostretch=0

while [[ $# -gt 0 ]]; do
  case $1 in
    --use-raw)
        use_raw=1
        shift
        ;;
    --nostretch)
        nostretch=1
        shift
        ;;
    -*)
        echo "Unknown option $1"
        exit 1;
        ;;
    esac
done

mkdir -p $outdir

OLDIFS=$IFS
IFS=$'\n'

alignment_files=$(find $(realpath $indir) -name "audio_alignments.txt")
echo "$alignment_files"

function extract_audio_at() {
    local line=$1
    local alignment_file=$2
    local audio_files=$3
    local start=$(echo $line | cut -f1); 
    local duration=$(echo $line | cut -f2); 

    if [ -z "$start" ] || [ -z "$duration" ]; then
        echo "Couldn't find start or duration for $alignment_file" && exit;
    fi

    local transcript=$(echo $line | cut -f3);
    local len=$(echo "$transcript" | wc -m)
    if [ $len -lt 2 ]; then
        echo "Error processing $alignment_file";
        exit -1;
    fi
    local bs_data=$(tail -n+2 $blendshapes | awk -F'[:,]' -v start=$start -v duration=$duration -F':' '{
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
    echo "Timings extracted, extracting audio"
    for audio_file in $audio_files; do
        local audio_id=$(basename $audio_file | sed "s/\.wav//g")
        echo "$transcript" > $outdir/"${audio_id}_$i".txt
        ffmpeg -n -i $audio_file -ss $start -t $duration -ar 16000 "$outdir/${audio_id}_$i.wav" 2>/dev/null 
        echo "$header" > "$outdir/${audio_id}_$i.csv"
        echo "$bs_data" >> "$outdir/${audio_id}_$i.csv"; 
        if [ $(wc -l "$outdir/${audio_id}_$i.csv" | cut -d' ' -f1) -lt 2 ]; then
            echo "Error handling blendshapes from $blendshapes";
        fi
    done
    #echo "Audio extracted"
}

# in every directory, we look for a single file called audio_alignments.txt
# this contains the start/stop times and accompanying transcript for an audio recording
for alignment_file in $alignment_files; do
    current=$(dirname $alignment_file)
 
    # find all audio files in directory
    audio_files=$(find $current -name "*.wav" -not -name "*.*.wav")
  
    if [ ! $nostretch ]; then
        # pitch-shift 
        for audio_file in $audio_files; do
            for ratio in 0.7 0.8 0.9 1.1 1.2 1.3; do
                ffmpeg -n -i $audio_file -af "asetrate=44100*$ratio, aresample=44100, atempo=1/$ratio" $(echo "$audio_file" | sed "s/\.wav/$ratio.wav/g")
            done
        done
        audio_files=$(find $current -name "*.wav")
    else
        audio_files=$(find $current -name "*.wav" -not -name "*.*.wav")
    fi
   

    if [ -z "$audio_files" ]; then
    	echo "No audio file matching $alignment_file" && exit -1;
    fi
    
    if [ ! $use_raw ]; then
        # make sure we use the calibrated blendshapes if available
        echo "use_raw "
        blendshapes=$(find $current -name "*_cal.csv")
    else
        blendshapes=$(find $current -name "*_raw.csv")
        # XX_raw.csv won't exist if the blendshapes weren't specifically recorded with the calibration flag, so just look for any CSV in the directory
        if [ -z "$blendshapes" ]; then
            blendshapes=$(find $current -name "*.csv");
        fi
    fi

    audio_alignments=$(cat $alignment_file | awk --field-separator=$'\t' '{printf "%s\t%f\t%s\n",$1,($2-$1),$3}' | sed -E "/^[[:space:]]$/d");
    
    
    header=$(head -n1 $blendshapes)
    echo "Generating blendshape CSV for $alignment_file";

    i=0
    # each line in the alignments file corresponds to a single phrase
    # for each line, extract the start/end time, find the matching period in the CSV and output to a new CSV
    for line in $audio_alignments; do
        if [ -z "$line" ]; then
            continue;
        fi 
        extract_audio_at "$line" "$alignment_file" "$audio_files"&
        i=$((i+1))
        if [ `expr $i % 5` -eq 0 ]; then
            wait
        fi
    done
    wait
done
IFS=$OLDIFS

