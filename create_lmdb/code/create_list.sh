# !/bin/bash

# This file is used to create the list of train and test files for training and testing procedures. After the run trainval.txt, test.txt and test_name_size.txt files will be generated. These files map each image to its label file.

# Modify the 'data_root_dir' to the location where your Images and Labels folders exist.
data_root_dir=/media/vkchcp0013/mstu_hpat/priyal/create_lmdb/Dataset/  # Modify the path with your folder having Images and Labels directories.

current_dir=`pwd`
echo "current_dir: "${current_dir}
dst_all_tmp=${current_dir}"/all_tmp.txt"
dst_file_trainval=${current_dir}"/trainval.txt"
dst_file_test=${current_dir}"/test.txt"
dst_file_test_name_size=${current_dir}"/test_name_size.txt"

length_imgs=`ls -l ${data_root_dir}/Images|grep '^-'|wc -l`
length_labels=`ls -l ${data_root_dir}/Labels|grep '^-'|wc -l`
echo "all images count: "${length_imgs}
echo "all labels count: "${length_labels}
if [ ${length_imgs} != ${length_labels} ]; then
	echo "Images and Labels not equal count. Images and Labels must be same count!"
else
    j=0
	for img in `ls ${data_root_dir}/Images|sort -h`
	do
		img_list[${j}]=${img}
		((j++))
	done

	k=0
	for label in `ls ${data_root_dir}/Labels|sort -h`
	do
		label_list[${k}]=${label}
		((k++))
	done
	
	for ((i=1;i<${length_imgs};i++))
	do
		left=${img_list[i]}
		right=${label_list[i]}
		content="Images/"${left}" Labels/"${right}
		echo ${content} >> ${dst_all_tmp}
		
	done
fi

# random shuffle the lines in all images
arr=(`seq ${length_imgs}`)
for ((i=0;i<10000;i++))
do
        let "a=$RANDOM%${length_imgs}"
        let "b=$RANDOM%${length_imgs}"
        tmp=${arr[$a]}
        arr[$a]=${arr[$b]}
        arr[$b]=$tmp
done

# change this value to split trainval and test, default is 0.8
split_ratio=0.8
boundry=`echo | awk "{print int(${length_imgs}*${split_ratio})}"`
echo "trainval count: "${boundry}
for i in ${arr[@]:0:${boundry}}
do
	sed -n "${i}p" ${dst_all_tmp} >> ${dst_file_trainval}
done

# generate test.txt and test_name_size.txt
for i in ${arr[@]:${boundry}:((${length_imgs}-${boundry}))}
do
	line=`sed -n -e "${i}p" ${dst_all_tmp}|cut -d ' ' -f 1`
	size=`identify ${data_root_dir}${line}|cut -d ' ' -f 3|sed -e "s/x/ /" | sed -r 's/([^ ]+) (.*)/\2 \1/'`
	echo ${line}
	name=`basename ${line} .png`
	echo ${name}" "${size} >> ${dst_file_test_name_size}
	sed -n "${i}p" ${dst_all_tmp} >> ${dst_file_test}
done

rm -f ${dst_all_tmp}


echo "Done!"


