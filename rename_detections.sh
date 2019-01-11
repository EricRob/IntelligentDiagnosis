## Only replace spaces in all .txt files
# for f in *.txt; do mv -- "$f" "${f// /_}"; done

# Append ".txt" to all files and replace " " with "_"
for file in *; do
	if [[ $file == *.txt ]] || [[ $file == *.sh ]]
	then echo "skipping $file"
	else
	mv -- "$file" "${file// /_}.txt"
	#	mv "$file" "${file}.txt"
	fi
	    
done
