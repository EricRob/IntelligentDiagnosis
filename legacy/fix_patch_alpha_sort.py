import os
import os.path

# takes patch names and pads the given index with zeros so the index will
# have 5 digits and will not require a "human sort"

#find nth occurence of specified character
def find_nth(name, str_to_find, n):
    start = name.find(str_to_find)
    while start >= 0 and n > 1:
        start = name.find(str_to_find, start+len(str_to_find))
        n -= 1
    return start


# pad filename with zeros so it will sort properly
def pad_zeros(name, n):
	if name.count("_") == 5: n = n - 1 # image was not cropped, contains one fewer "_"
	start = find_nth(name, "_", n ) + 1 # start of index
	end = find_nth(name, "_", n + 1 ) #end of index
	if end == -1:
		end = name.find('.') # no trailing "_", use "." from file suffix

	x_index = name[start:end]
	
	if len(x_index) == 3 :
		x_index = '00' + x_index
	elif len(x_index) == 4 :
		x_index = '0' + x_index



	new_name = name[0:start] + x_index + name[end:]
	return new_name

lst = os.listdir('.') 

for dirpath, dirnames, filenames in os.walk("."):
	for filename in [f for f in filenames if f.endswith(".tif")]:
		new_name = pad_zeros(filename, 5)
		new_name = pad_zeros(new_name, 6)
		pathname = os.path.join(dirpath, filename)
		new_pathname = os.path.join(dirpath, new_name)
		print(new_pathname)
		os.rename(pathname, new_pathname)
