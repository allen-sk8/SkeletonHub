## !/bin/bash
## environment: carl
## example usage:
## find /home/c1l1mo/projects/mislab_basketball/visualization -name "*.mp4" -exec ./change_encoding.sh {} \;
## find /home/nadiateng2005/running-speed-estimator-1 -name "*.mp4" -exec /home/allen/change_encoding.sh {} \;
## find /home/allen/SkateApp_web/frontend/public/std_vids -name "*.mp4" -exec /home/allen/change_encoding.sh {} \;
## find /home/jimmy_1018/coachme/dataset/skeleton/pushup -name "*.mp4" -exec /home/allen/change_encoding.sh {} \;


##current directory:
current_dir=$(pwd)


cd /home/allen
SOURCE=$1

if [ -z "$SOURCE" ]
then
    echo "No source provided"
    exit 1
fi

## use carl environment
#source /home/c1l1mo/miniconda3/etc/profile.d/conda.sh
#conda activate carl


SUBSTRING="${SOURCE%/*}"
OUTPUT="${SUBSTRING}/output.mp4"


ffmpeg -i $SOURCE -c:v libx264 -crf 23 -c:a copy $OUTPUT

mv $OUTPUT $SOURCE

cd $current_dir