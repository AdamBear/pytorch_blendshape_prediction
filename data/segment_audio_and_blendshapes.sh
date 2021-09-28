indir=$1
outdir=$2
if [ $# -lt 2 ] ; then
	echo -e "Usage: ./segment.sh <indir> <outdir>\nFinds every .wav file in the provided <indir>, looks for a file named audio_alignments.txt in the same directory,\nnormalizes the viseme CSV according to the alignments file and copies the audio, transcript and viseme file to <outdir>" && exit -1;
fi

OLDIFS=$IFS
IFS=$'\n'

alignment_files=$(find $(realpath $indir) -name "audio_alignments.txt")
echo "$alignment_files"
i=0; 
for alignment_file in $alignment_files; do
    current=$(dirname $alignment_file)

    audio_file=$(find $current -name "*.wav")

    if [ -z "$audio_file" ]; then
    	echo "No audio file matching $alignment_file" && exit -1;
    fi
    
    blendshapes=$(echo "$audio_file" | sed "s/wav/csv/g")

    audio_alignments=$(cat $alignment_file | awk --field-separator=$'\t' '{printf "%s\t%f\t%s\n",$1,($2-$1),$3}' | sed -E "/^[[:space:]]$/d");
    echo "Generating blendshape CSV for $alignment_file";
    header=$(head -n1 $blendshapes)

    # each line in the alignments file corresponds to a single phrase
    # for each line, extract the start/end time, find the matching period in the CSV and output to a new CSV
    for line in $audio_alignments; do
        if [ -z "$line" ]; then
            continue;
        fi 
        start=$(echo $line | cut -f1); 
        duration=$(echo $line | cut -f2); 
        transcript=$(echo $line | cut -f3);
        len=$(echo "$transcript" | wc -m)
        if [ $len -lt 2 ]; then
            echo "Error processing $alignment_file";
            exit -1;
        fi
        echo "$transcript" > $outdir/$i.txt
        ffmpeg -y -i $audio_file -ss $start -t $duration $outdir/$i.wav 2>/dev/null 
        echo "$header" > $outdir/$i.csv
        if [ -z "$start" ] || [ -z "$duration" ]; then
            echo "Couldn't find start or duration for $alignment_file" && exit;
        fi

        tail -n+2 $blendshapes | awk -F'[:,]' -v start=$start -v duration=$duration -F':' '{
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
        }' >> $outdir/$i.csv; 
        if [ $(wc -l $outdir/$i.csv | cut -d' ' -f1) -lt 2 ]; then
            echo "Error handling blendshapes from $blendshapes";
        fi
        i=$((i+1)); 
    done
done
IFS=$OLDIFS

