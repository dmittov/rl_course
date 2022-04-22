#!/bin/bash

git diff --name-only HEAD^ HEAD -- . :^.github | sort | uniq > files.txt
dir=''
while IFS= read -r file
do
  parent_dir=$(dirname -- "$file")
  if [[ " ${parent_dir} " =~ week* ]]; then
    dir_name=$(echo "$parent_dir" | cut -d/ -f1-1)
    echo "::notice title=${dir_name} Changed::Updated file ${file}"
    if ! [[ $dir_name == "." ]] && [[ -d $dir_name ]]; then
      if [[ -z $dir ]]; then
        dir="{\"dir\":\"$dir_name\"}"
      else
        if ! [[ $dir == *$dir_name* ]]; then
          dir="$dir, {\"dir\":\"$dir_name\"}"
        fi
      fi
    fi
  fi
done < files.txt
dir="{\"include\":[$dir]}"
echo "::notice title=Updated Directories::${dir}"
echo "::set-output name=matrix::$dir"