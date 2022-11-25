for file in `ls $1` ; do
    
    echo "mv -v $1/"$file" $1/"${file#*41--}""
    mv -v $1/"$file" $1/"${file#*41--}"
done
