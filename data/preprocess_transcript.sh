APP_COMPONENTS_DIR=$HOME/projects/polyvox/polyvox_framework

for dir in training test; do
    transcripts=$(find $(realpath $dir) -name "*.txt" -not -name "*_pp.txt")
    echo "$transcripts" | wc -l
    
    as_ipa=$(cat $transcripts | dart $APP_COMPONENTS_DIR/lib/scripts/dictionary/text_to_ipa.dart --lexicon=$APP_COMPONENTS_DIR/assets/lexicon_full.txt --symbol_ids=$APP_COMPONENTS_DIR/assets/symbol_ids.txt)
    i=0;
    num_transcripts=$(echo "$transcripts" | wc -l)
    
    readarray -t transcripts <<< "$transcripts";
    num_ipa=$(echo "$as_ipa" | wc -l)
    if [ $num_ipa -ne $num_transcripts ]; then
        echo "Mismatch between number of raw transcripts ($num_transcripts) and number of lines from IPA converter output ($num_ipa)";
        exit -1;
    fi
        
    OLDIFS=$IFS
    IFS=$'\n'
    for line in $as_ipa; do
        echo $line > $(echo "${transcripts[i]}" | sed "s/\.txt/_pp.txt/")
        i=$((i+1))
    done
    IFS=$OLDIFS
done 
    
