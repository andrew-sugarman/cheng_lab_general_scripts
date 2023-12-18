#!/bin/bash
start=`date +%s`
sample_name=${PWD##*/}
sample_name=${PWD##*/}
echo $sample_name
# Absolute path this script is in. /home/user/bin
# SCRIPT=$(readlink -f $0) #FAILS BECAUSE SLURM SPOOL
SCRIPT="${PWD}/"
SCRIPTPATH=`dirname "$SCRIPT"`
Recon_script_path="${PWD}/R2.54_Berkeley8_hpc_v1.70.py"
MIP_script_path="${PWD}/gen_mip_hpc_v1.py"
Ortho_crop_script_path="${PWD}/otho_crop_hpc_v0.py"
echo $SCRIPT
echo $SCRIPTPATH
echo $Recon_script_path




reconstruct=true
target_nodes_reconstruct='dense'
# target_nodes_reconstruct='compute'
generate_MIP=false
target_nodes_mip='compute'
bit_level_crop=false
generate_n5=false
target_nodes_ortho_crop='dense'
Memfor_nodes_ortho_crop='1250G'
# target_nodes_ortho_crop='compute'
# Memfor_nodes_ortho_crop='120G'
ortho_crop=false


# cpu_per_node=31
# mem_per_node='30G'
# skew_angles=(-.01 -.025 -.05 .01 .025 .05)
# skew_angles=(-.01)
# skew_angle=-0.2661

mkdir -p -m2777 slurm_logs

# rm -r /tmp/VANSELOW_BABYYY/

if $reconstruct
	then
		sbatch -p $target_nodes_reconstruct -A lab_cheng --exclusive --job-name=aiw  -W  -o ./slurm_logs/aiw.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; source activate recon; time python -u '$Recon_script_path'" &
	wait
fi
# rm -r /tmp/VANSELOW_BABYYY/
chmod -R 2777 ./gains/
chmod -R 2777 ./slurm_logs/

if $generate_MIP
	then
		sbatch -p $target_nodes_mip -A lab_cheng --cpus-per-task=20 --mem=20G --job-name=aiw_mip  -W  -o ./slurm_logs/aiw_mip.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; source activate recon; time python -u '$MIP_script_path'" &
	wait
fi

if $ortho_crop
	then
		# sbatch -p $target_nodes_ortho_crop  -A lab_cheng --exclusive --exclude=psh01com1hdns01,psh01com1hdns03 --job-name=aiw_crop1  -W  -o ./slurm_logs/aiw_crop1.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; source activate recon2; time python -u $Ortho_crop_script_path" &
		sbatch -p $target_nodes_ortho_crop  -A lab_cheng --cpus-per-task=46 --mem=Memfor_nodes_ortho_crop --exclude=psh01com1hdns01,psh01com1hdns03,psh01com1hdns04 --job-name=aiw_crop1  -W  -o ./slurm_logs/aiw_crop1.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; source activate recon; time python -u '$Ortho_crop_script_path'" &
	wait
fi
#  --exclude=psh01com1hdns03,psh01com1hdns01,psh01com1hdns03,psh01com1hdns04
end=`date +%s`
runtime=$((end-start))
echo 'full run time: $runtime' 

# echo 'removing tmp folder'
# rm -r /tmp/VANSELOW_BABYYY


# for i in "${!skew_angles[@]}";
	# do
		# # sbatch -p $target_nodes --exclusive --job-name=aiw_1  -W  -o slurm_logs/aiw_1.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; source activate recon2; time python -u /gpfs/Labs/Cheng/phenome/hpc_scripting/reconstruction/allinwonderful_v2.py $SCRIPT $skew_angle" &
		# sbatch -p $target_nodes --cpus-per-task=$cpu_per_node --mem=$mem_per_node --job-name=aiw_$i  -W  -o slurm_logs/aiw_$i.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; source activate recon2; time python -u /gpfs/Labs/Cheng/phenome/hpc_scripting/reconstruction/allinwonderful_v2.1.0.py $SCRIPT ${skew_angles[$i]}" &
		# sleep 25  
	# done	
	# wait

# end=`date +%s`
# runtime=$((end-start))
# echo $runtime






# #maybe filter out psh01com1hcom26???? 4/1/2022
# debug='debug'
# mix_nodes=true

# #Channel combining to prep for Ilastik classification
# nodes_to_use_composite=10


# #Ilastik classification and output collpase
# nodes_to_use_ilastik=55


# nodes_to_use_combine_probabilities=10
# tubemap_chunk_overlap_size=30
# overlap_x=500

# sixteen_bit_lower_thresh=5500 #ilastik with fill classifier
# filling_thresh=.5
# resample_factor_for_filling=2

# x_split=2 #DO NOT CHANGE FOR NOW.

# # nodes_to_use_binarize=20
# nodes_to_use_binarize=`expr $nodes_to_use_combine_probabilities \* $x_split`
# # nodes_to_use_smoothing=5


# combine_channels=false
# ilastik_apply=false
# combine_probabilities=false

# tubemap_binarize=false
# register=true
# combine_skele=false
# tubemap_graph=false #40 minutes skel to graph on gpu node
# post_hoc_anal=false


# tubemap_smooth=false

# chmod_stuff=false

# sample_name=${PWD##*/}
# echo $sample_name
# # Absolute path this script is in. /home/user/bin
# # SCRIPT=$(readlink -f $0) #FAILS BECAUSE SLURM SPOOL
# SCRIPT="${PWD}/"
# SCRIPTPATH=`dirname $SCRIPT`
# echo $SCRIPT
# echo $SCRIPTPATH




# total_files_ch00=$(find ./stitched_00 -maxdepth 1 -type f -name '*.tif' -printf x | wc -c)



# echo $total_files_ch00



# mkdir -p -m2777 slurm_logs

# #combining channel 00 01 02:
# if $combine_channels;
	# then
		# for i in `seq 1 $nodes_to_use_composite`;
		# do
			# if $mix_nodes;
			# then
				# sbatch -p $target_nodes  --cpus-per-task=11 --mem=16G  --job-name=cmbC_$i  -W  -o slurm_logs/combine_channels_$i.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; source activate ClearMap ; time python -u /gpfs/Labs/Kim/TubeMap_data/scripts/channel_combine_sh_target.py $nodes_to_use_composite $i $debug $SCRIPT" &
			# else
				# sbatch -p $target_nodes --exclusive --job-name=cmbC_$i  -W  -o slurm_logs/combine_channels_$i.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; source activate ClearMap ; time python -u /gpfs/Labs/Kim/TubeMap_data/scripts/channel_combine_sh_target.py $nodes_to_use_composite $i $debug $SCRIPT" &
			# fi
			# # sbatch -p compute --job-name=cmb_$i  -W  -o slurm_logs/combine_channels_$i.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; source activate ClearMap ; time python -u /gpfs/Labs/Kim/TubeMap_data/scripts/channel_combine_hdf5_sh_target.py $nodes_to_use_composite $i  $tubemap_chunk_overlap_size $SCRIPT" &
			# # sbatch -p compute --exclusive --job-name=cmb_$i  -W  -o slurm_logs/combine_channels_$i.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; source activate ClearMap ; time python -u /gpfs/Labs/Kim/TubeMap_data/scripts/channel_combine_3dtiff_sh_target.py $nodes_to_use_composite $i  $tubemap_chunk_overlap_size $SCRIPT" &
			# sleep 1
		# done
		# wait
		# chmod -R 2777 ./tubemap/
		# chmod -R 2777 ./slurm_logs/
# fi



# if $ilastik_apply;
	# then
		# for i in `seq 1 $nodes_to_use_ilastik`;
		# do
			# # sbatch -p compute --exclusive --job-name=ilsk_$i  -W  -o slurm_logs/ilastik_$i.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; module load ilastik/1.3.3 ; source activate ClearMap ; time python -u /gpfs/Labs/Kim/TubeMap_data/scripts/ilastik_apply_chunk_sh_target_composite_3dtiff.py $nodes_to_use_ilastik $i $SCRIPT" &
			# # sbatch -p compute --job-name=ilsk_$i  -W  -o slurm_logs/ilastik_$i.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; module load ilastik/1.3.3 ; source activate ClearMap ; time python -u /gpfs/Labs/Kim/TubeMap_data/scripts/ilastik_apply_chunk_sh_target_composite_hdf5.py $nodes_to_use_ilastik $i $SCRIPT" &
			# if $mix_nodes;
			# then
				# echo $i
				# sbatch -p $target_nodes --cpus-per-task=15 --mem=42G --job-name=ilsk_$i  -W -o slurm_logs/ilastik_$i.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; module load ilastik/1.3.3 ; source activate ClearMap ; unset PYTHONPATH; time python -u /gpfs/Labs/Kim/TubeMap_data/scripts/ilastik_apply_chunk_sh_target_composite.py $nodes_to_use_ilastik $i $debug $SCRIPT" &
			# else
				# sbatch -p $target_nodes --exclusive --job-name=ilsk_$i  -W  -o slurm_logs/ilastik_$i.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; module load ilastik/1.3.3 ; source activate ClearMap ; unset PYTHONPATH; time python -u /gpfs/Labs/Kim/TubeMap_data/scripts/ilastik_apply_chunk_sh_target_composite.py $nodes_to_use_ilastik $i $debug $SCRIPT" &
			# fi
			# sleep .5
		# done
		# wait
		# chmod -R 2777 ./tubemap/
		# chmod -R 2777 ./slurm_logs/

# fi



# if $combine_probabilities;
	# then
		# for i in `seq 1 $nodes_to_use_combine_probabilities`;
		# do
			# # if $mix_nodes
			# # sbatch -p compute --exclusive --job-name=cmbP_$i  -W  -o slurm_logs/combine_probabilities_$i.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; source activate ClearMap ; time python -u /gpfs/Labs/Kim/TubeMap_data/scripts/probability_combine_sh_target.py $nodes_to_use_combine_probabilities $i $debug $SCRIPT" &
			# if $mix_nodes;
			# then
				# # sbatch -p compute --cpus-per-task=10 --mem=16G  --job-name=cmbP_$i  -W  -o slurm_logs/combine_probabilities_$i.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; source activate ClearMap ; time python -u /gpfs/Labs/Kim/TubeMap_data/scripts/probability_combine_sh_target.py $nodes_to_use_combine_probabilities $i $overlap_x $debug $sixteen_bit_lower_thresh $SCRIPT" &
				# sbatch -p $target_nodes --cpus-per-task=10 --mem=16G  --job-name=cmbP_$i  -W  -o slurm_logs/combine_probabilities_$i.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; source activate ClearMap ; time python -u /gpfs/Labs/Kim/TubeMap_data/scripts/simple_segmentation_combine_sh_target.py $nodes_to_use_combine_probabilities $i $overlap_x $debug $sixteen_bit_lower_thresh $SCRIPT" &
			
			# else
				# sbatch -p $target_nodes --exclusive --job-name=cmbP_$i  -W  -o slurm_logs/combine_probabilities_$i.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; source activate ClearMap ; time python -u /gpfs/Labs/Kim/TubeMap_data/scripts/probability_combine_sh_target.py $nodes_to_use_combine_probabilities $i $overlap_x $debug $sixteen_bit_lower_thresh $SCRIPT" &
			# fi
			# sleep 1
		# done
		# wait
		# chmod -R 2777 ./tubemap/
		# chmod -R 2777 ./slurm_logs/
# fi

# #--exclude=psh01com1ocom02,psh01com1ocom05,psh01com1ocom04,psh01com1ocom01,psh01com1ocom08,psh01com1ocom06,psh01com1hcom03,psh01com1hcom15,psh01com1ocom07,psh01com1ochm01,psh01com1ocom03
# #--exclude=psh01com1ocom06,psh01com1hcom15
# #--exclude=psh01com1hcom14 
# #--exclude=psh01com1hcom28
# # --requeue

# #/usr/bin/xvfb-run: line 181: 22199 Bus error			xvfb-run -d <--- runs but ometimes fails  try xvfb-run -a again???
# #xvfb  '-screen 0, 1280x1024x24'

# # || scontrol requeue $SLURM_JOB_ID
# mix_nodes=true
# if $tubemap_binarize;
	# then
		# for i in `seq 1 $nodes_to_use_binarize`;
		# # for i in {3,4,11,17,21,27,36,38};
		# # for i in {10,10};
		# do
			# if $mix_nodes;
			# then
				# echo $i
				# #channel 00
				# input_dir='ilastik_binary_ch00'
				# output_dir='ilastik_binary_ch00_filled'
				# sbatch -p $target_nodes --requeue --exclude=psh01com1hcom28,psh01com1hcom29,psh01com1hcom30,psh01com1hcom32,psh01com1hcom31,psh01com1hcom02,psh01com1hcom04,dragen --cpus-per-task=10 --mem=10G --job-name=fl_0_$i  -W  -o slurm_logs/fill_skel_0_$i.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; source activate ClearMap ; time python -u  /gpfs/Labs/Kim/TubeMap_data/scripts/TubeMap_dan_v3.1_sh_target.py $nodes_to_use_binarize $i $tubemap_chunk_overlap_size $SCRIPT binarization $input_dir $output_dir $overlap_x $sixteen_bit_lower_thresh $filling_thresh $resample_factor_for_filling $debug" &
				# sleep 1
				# #channel 01
				# input_dir='ilastik_binary_ch01'
				# output_dir='ilastik_binary_ch01_filled'
				# sbatch -p $target_nodes --requeue --exclude=psh01com1hcom28,psh01com1hcom29,psh01com1hcom30,psh01com1hcom32,psh01com1hcom31,psh01com1hcom02,psh01com1hcom04,dragen --cpus-per-task=10 --mem=10G --job-name=fl_1_$i  -W  -o slurm_logs/fill_skel_1_$i.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; source activate ClearMap ; time  python -u  /gpfs/Labs/Kim/TubeMap_data/scripts/TubeMap_dan_v3.1_sh_target.py $nodes_to_use_binarize $i $tubemap_chunk_overlap_size $SCRIPT binarization $input_dir $output_dir $overlap_x $sixteen_bit_lower_thresh $filling_thresh $resample_factor_for_filling $debug" &
				# sleep 1
				# #combine channels
				# input_dir='ilastik_binary_ch00'
				# input_dir_1='ilastik_binary_ch01'
				# output_dir='ilastik_binary_combined_filled' 
				# sbatch -p $target_nodes --requeue --exclude=psh01com1hcom28,psh01com1hcom29,psh01com1hcom30,psh01com1hcom32,psh01com1hcom31,psh01com1hcom02,psh01com1hcom04,dragen --cpus-per-task=10 --mem=10G --job-name=fl_01_$i  -W  -o slurm_logs/fill_skel_2_$i.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; source activate ClearMap ; time  python -u  /gpfs/Labs/Kim/TubeMap_data/scripts/TubeMap_dan_v3.1_sh_target.py $nodes_to_use_binarize $i $tubemap_chunk_overlap_size $SCRIPT binarization $input_dir $output_dir $overlap_x $sixteen_bit_lower_thresh $filling_thresh $resample_factor_for_filling $input_dir_1 $debug" &
				# sleep 2

			# else
				# echo $i
				# #channel 00
				# input_dir='ilastik_binary_ch00'
				# output_dir='ilastik_binary_ch00_filled'
				# sbatch -p $target_nodes --requeue --exclude=psh01com1hcom28 --exclusive --job-name=fl_0_$i  -W  -o slurm_logs/fill_skel_0_$i.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; source activate ClearMap ; time python -u  /gpfs/Labs/Kim/TubeMap_data/scripts/TubeMap_dan_v3.1_sh_target.py $nodes_to_use_binarize $i $tubemap_chunk_overlap_size $SCRIPT binarization $input_dir $output_dir $overlap_x $sixteen_bit_lower_thresh $filling_thresh $resample_factor_for_filling $debug" &
				# sleep 1
				# #channel 01
				# input_dir='ilastik_binary_ch01'
				# output_dir='ilastik_binary_ch01_filled'
				# sbatch -p $target_nodes --requeue --exclude=psh01com1hcom28 --exclusive --job-name=fl_1_$i  -W  -o slurm_logs/fill_skel_1_$i.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; source activate ClearMap ; time  python -u  /gpfs/Labs/Kim/TubeMap_data/scripts/TubeMap_dan_v3.1_sh_target.py $nodes_to_use_binarize $i $tubemap_chunk_overlap_size $SCRIPT binarization $input_dir $output_dir $overlap_x $sixteen_bit_lower_thresh $filling_thresh $resample_factor_for_filling $debug" &
				# sleep 1
				# #combine channels
				# input_dir='ilastik_binary_ch00'
				# input_dir_1='ilastik_binary_ch01'
				# output_dir='ilastik_binary_combined_filled' 
				# sbatch -p $target_nodes --requeue --exclude=psh01com1hcom28 --exclusive --job-name=fl_01_$i  -W  -o slurm_logs/fill_skel_2_$i.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; source activate ClearMap ; time  python -u  /gpfs/Labs/Kim/TubeMap_data/scripts/TubeMap_dan_v3.1_sh_target.py $nodes_to_use_binarize $i $tubemap_chunk_overlap_size $SCRIPT binarization $input_dir $output_dir $overlap_x $sixteen_bit_lower_thresh $filling_thresh $resample_factor_for_filling $input_dir_1 $debug" &
				# sleep 2
				# # grep --include=\*fill_* -rnwl './slurm_logs' -e "core dumped"
			# fi
		# done
		
		
		# wait
		# chmod -R 2777 ./tubemap/ilastik_binary_ch00
		# chmod -R 2777 ./tubemap/ilastik_binary_ch01
		# chmod -R 2777 ./tubemap/ilastik_binary_combined_filled
		# chmod -R 2777 ./slurm_logs/
		
		
		# # #combine channel 00 01
		# # input_dir='ilastik_binary_ch00_filled_debug_half_1_filled'
		# # input_dir_1='ilastik_binary_ch01_filled_debug_half_1_filled'
		# # output_dir='ilastik_binary_filled_half_1'
		# # sbatch -p $target_nodes --requeue --cpus-per-task=10 --mem=28G --job-name=fl_0_$i  -W  -o slurm_logs/fill_skel_0_$i.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; source activate ClearMap ; time  xvfb-run --auto-servernum python -u  /gpfs/Labs/Kim/TubeMap_data/scripts/TubeMap_dan_v3.1_sh_target.py $nodes_to_use_binarize $i $tubemap_chunk_overlap_size $SCRIPT combine_and_skel $input_dir $output_dir $overlap_x $sixteen_bit_lower_thresh $filling_thresh $resample_factor_for_filling $input_dir_1 $debug" &
		# # input_dir='ilastik_binary_ch00_filled_debug_half_2_filled'
		# # input_dir_1='ilastik_binary_ch01_filled_debug_half_2_filled'
		# # output_dir='ilastik_binary_filled_half_2'
		# # sbatch -p $target_nodes --requeue --cpus-per-task=10 --mem=28G --job-name=fl_0_$i  -W  -o slurm_logs/fill_skel_0_$i.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; source activate ClearMap ; time  xvfb-run --auto-servernum python -u  /gpfs/Labs/Kim/TubeMap_data/scripts/TubeMap_dan_v3.1_sh_target.py $nodes_to_use_binarize $i $tubemap_chunk_overlap_size $SCRIPT combine_and_skel $input_dir $output_dir $overlap_x $sixteen_bit_lower_thresh $filling_thresh $resample_factor_for_filling $input_dir_1 $debug" &
		
		
		# wait
		
# fi
# mix_nodes=true

# if $register;
	# then
	# i=0
	# input_dir='ilastik_binary_ch00'
	# output_dir='ilastik_binary_ch00_filled'
	# #--cpus-per-task=10 --mem=28G
	# sbatch -p compute --exclusive --requeue --job-name=regi_$i -W -o slurm_logs/register_$i.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; source activate ClearMap ; time  xvfb-run --auto-servernum python -u  /gpfs/Labs/Kim/TubeMap_data/scripts/TubeMap_dan_v3.1_sh_target.py $nodes_to_use_binarize $i $tubemap_chunk_overlap_size $SCRIPT registration $input_dir $output_dir $overlap_x $sixteen_bit_lower_thresh $filling_thresh $resample_factor_for_filling $debug" 
	
# fi
# # if $tubemap_binarize;
	# # then
		# # for i in `seq 1 $nodes_to_use_binarize`;
		# # # for i in {3,4,11,17,21,27,36,38};
		# # # for i in {10,10};
		# # do
			# # if $mix_nodes;
			# # then
				# # #channel 00
				# # input_dir='ilastik_binary_ch00'
				# # output_dir='ilastik_binary_ch00_filled'
				# # sbatch -p $target_nodes --exclude=psh01com1hcom14,psh01com1ocom01 --requeue --cpus-per-task=14 --mem=36G --job-name=fl_0_$i  -W  -o slurm_logs/fill_skel_0_$i.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; source activate ClearMap ; time xvfb-run -w $(( ( RANDOM % 60 ) + 1 )) --auto-servernum python -u  /gpfs/Labs/Kim/TubeMap_data/scripts/TubeMap_dan_v3.1_sh_target.py $nodes_to_use_binarize $i $tubemap_chunk_overlap_size $SCRIPT binarization $input_dir $output_dir $overlap_x $sixteen_bit_lower_thresh $filling_thresh $resample_factor_for_filling $debug" &
				# # sleep 5
				# # #channel 01
				# # input_dir='ilastik_binary_ch01'
				# # output_dir='ilastik_binary_ch01_filled'
				# # sbatch -p $target_nodes --exclude=psh01com1hcom14,psh01com1ocom01 --requeue --cpus-per-task=14 --mem=36G --job-name=fl_1_$i  -W  -o slurm_logs/fill_skel_1_$i.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; source activate ClearMap ; time xvfb-run -w $(( ( RANDOM % 60 ) + 1 )) --auto-servernum python -u  /gpfs/Labs/Kim/TubeMap_data/scripts/TubeMap_dan_v3.1_sh_target.py $nodes_to_use_binarize $i $tubemap_chunk_overlap_size $SCRIPT binarization $input_dir $output_dir $overlap_x $sixteen_bit_lower_thresh $filling_thresh $resample_factor_for_filling $debug" &
				# # sleep 5
				# # #combine channels
				# # input_dir='ilastik_binary_ch00'
				# # input_dir_1='ilastik_binary_ch01'
				# # output_dir='ilastik_binary_combined_filled' 
				# # sbatch -p $target_nodes --exclude=psh01com1hcom14,psh01com1ocom01 --requeue --cpus-per-task=14 --mem=36G --job-name=fl_01_$i  -W  -o slurm_logs/fill_skel_combined_$i.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; source activate ClearMap ; time xvfb-run -w $(( ( RANDOM % 60 ) + 1 )) --auto-servernum python -u  /gpfs/Labs/Kim/TubeMap_data/scripts/TubeMap_dan_v3.1_sh_target.py $nodes_to_use_binarize $i $tubemap_chunk_overlap_size $SCRIPT binarization $input_dir $output_dir $overlap_x $sixteen_bit_lower_thresh $filling_thresh $resample_factor_for_filling $input_dir_1 $debug" &
		

			# # else
				# # sbatch -p $target_nodes --requeue --exclusive --job-name=bniz_$i  -W  -o slurm_logs/binarize_$i.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; source activate ClearMap ; time  xvfb-run -d -a python -u  /gpfs/Labs/Kim/TubeMap_data/scripts/TubeMap_dan_v3.1_sh_target.py $nodes_to_use_binarize $i $tubemap_chunk_overlap_size $SCRIPT binarization $input_dir tm_binarized $overlap_x $sixteen_bit_lower_thresh $filling_thresh $resample_factor_for_filling $debug" &
			# # fi
			# # sleep 5
		# # done
		# # wait
		# # chmod -R 2777 ./tubemap/ilastik_binary_ch00
		# # chmod -R 2777 ./tubemap/ilastik_binary_ch01
		# # chmod -R 2777 ./tubemap/ilastik_binary_combined_filled
		# # chmod -R 2777 ./slurm_logs/
		
		
		# # # #combine channel 00 01
		# # # input_dir='ilastik_binary_ch00_filled_debug_half_1_filled'
		# # # input_dir_1='ilastik_binary_ch01_filled_debug_half_1_filled'
		# # # output_dir='ilastik_binary_filled_half_1'
		# # # sbatch -p $target_nodes --requeue --cpus-per-task=10 --mem=28G --job-name=fl_0_$i  -W  -o slurm_logs/fill_skel_0_$i.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; source activate ClearMap ; time  xvfb-run --auto-servernum python -u  /gpfs/Labs/Kim/TubeMap_data/scripts/TubeMap_dan_v3.1_sh_target.py $nodes_to_use_binarize $i $tubemap_chunk_overlap_size $SCRIPT combine_and_skel $input_dir $output_dir $overlap_x $sixteen_bit_lower_thresh $filling_thresh $resample_factor_for_filling $input_dir_1 $debug" &
		# # # input_dir='ilastik_binary_ch00_filled_debug_half_2_filled'
		# # # input_dir_1='ilastik_binary_ch01_filled_debug_half_2_filled'
		# # # output_dir='ilastik_binary_filled_half_2'
		# # # sbatch -p $target_nodes --requeue --cpus-per-task=10 --mem=28G --job-name=fl_0_$i  -W  -o slurm_logs/fill_skel_0_$i.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; source activate ClearMap ; time  xvfb-run --auto-servernum python -u  /gpfs/Labs/Kim/TubeMap_data/scripts/TubeMap_dan_v3.1_sh_target.py $nodes_to_use_binarize $i $tubemap_chunk_overlap_size $SCRIPT combine_and_skel $input_dir $output_dir $overlap_x $sixteen_bit_lower_thresh $filling_thresh $resample_factor_for_filling $input_dir_1 $debug" &
		
		
		# # wait
		
# # fi





# if $combine_skele;
	# then
	# i=0
	# input_dir1='ilastik_binary_ch00_filled_debug_half_1_skeleton'
	# input_dir2='ilastik_binary_ch00_filled_debug_half_2_skeleton'
	# input_file_generic='ch00'
	# output_tiff_dir='skeleton_recombined_ch00'
	# combine_output_base='skeleton_recombined_ch00'
	# # if $combine_skele;
	# sbatch -p $target_nodes --cpus-per-task=20 --mem=40G  --job-name=cmbG_$i  -W  -o slurm_logs/combine_$i.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; source activate ClearMap ; time python -u /gpfs/Labs/Kim/TubeMap_data/scripts/generic_combine_sh_target.py $nodes_to_use_combine_probabilities $i $overlap_x $debug $sixteen_bit_lower_thresh $input_dir1 $input_dir2 $input_file_generic $combine_output_base  $total_files_ch00 $output_tiff_dir $SCRIPT" &
	
	# i=1
	# input_dir1='ilastik_binary_ch01_filled_debug_half_1_skeleton'
	# input_dir2='ilastik_binary_ch01_filled_debug_half_2_skeleton'
	# input_file_generic='ch01'
	# output_tiff_dir='skeleton_recombined_ch01'
	# combine_output_base='skeleton_recombined_ch01'
	# # if $combine_skele;
	# sbatch -p $target_nodes --cpus-per-task=20 --mem=40G  --job-name=cmbG_$i  -W  -o slurm_logs/combine_$i.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; source activate ClearMap ; time python -u /gpfs/Labs/Kim/TubeMap_data/scripts/generic_combine_sh_target.py $nodes_to_use_combine_probabilities $i $overlap_x $debug $sixteen_bit_lower_thresh $input_dir1 $input_dir2 $input_file_generic $combine_output_base  $total_files_ch00 $output_tiff_dir $SCRIPT" &
	
	# i=2
	# input_dir1='ilastik_binary_ch00_filled_debug_half_1_filled'
	# input_dir2='ilastik_binary_ch00_filled_debug_half_2_filled'
	# input_file_generic='ch00'
	# output_tiff_dir='filled_recombined_ch00'
	# combine_output_base='filled_recombined_ch00'
	# # if $combine_skele;
	# sbatch -p $target_nodes --cpus-per-task=20 --mem=40G  --job-name=cmbG_$i  -W  -o slurm_logs/combine_$i.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; source activate ClearMap ; time python -u /gpfs/Labs/Kim/TubeMap_data/scripts/generic_combine_sh_target.py $nodes_to_use_combine_probabilities $i $overlap_x $debug $sixteen_bit_lower_thresh $input_dir1 $input_dir2 $input_file_generic $combine_output_base  $total_files_ch00 $output_tiff_dir $SCRIPT" &
	
	# i=3
	# input_dir1='ilastik_binary_ch01_filled_debug_half_1_filled'
	# input_dir2='ilastik_binary_ch01_filled_debug_half_2_filled'
	# input_file_generic='ch01'
	# output_tiff_dir='filled_recombined_ch01'
	# combine_output_base='filled_recombined_ch01'
	# # if $combine_skele;
	# sbatch -p $target_nodes --cpus-per-task=20 --mem=40G  --job-name=cmbG_$i  -W  -o slurm_logs/combine_$i.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; source activate ClearMap ; time python -u /gpfs/Labs/Kim/TubeMap_data/scripts/generic_combine_sh_target.py $nodes_to_use_combine_probabilities $i $overlap_x $debug $sixteen_bit_lower_thresh $input_dir1 $input_dir2 $input_file_generic $combine_output_base  $total_files_ch00 $output_tiff_dir $SCRIPT" &
	
	# i=4
	# input_dir1='ilastik_binary_combined_filled_debug_half_1_filled'
	# input_dir2='ilastik_binary_combined_filled_debug_half_2_filled'
	# input_file_generic='ch00'
	# output_tiff_dir='filled_recombined_ch00ch01'
	# combine_output_base='filled_recombined_ch00ch01'
	# # if $combine_skele;
	# sbatch -p $target_nodes --cpus-per-task=20 --mem=40G  --job-name=cmbG_$i  -W  -o slurm_logs/combine_$i.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; source activate ClearMap ; time python -u /gpfs/Labs/Kim/TubeMap_data/scripts/generic_combine_sh_target.py $nodes_to_use_combine_probabilities $i $overlap_x $debug $sixteen_bit_lower_thresh $input_dir1 $input_dir2 $input_file_generic $combine_output_base  $total_files_ch00 $output_tiff_dir $SCRIPT" &

	# i=5
	# input_dir1='ilastik_binary_combined_filled_debug_half_1_skeleton'
	# input_dir2='ilastik_binary_combined_filled_debug_half_2_skeleton'
	# input_file_generic='ch00'
	# output_tiff_dir='skeleton_recombined_ch00ch01'
	# combine_output_base='skeleton_recombined_ch00ch01'
	# # if $combine_skele;
	# sbatch -p $target_nodes --cpus-per-task=20 --mem=40G  --job-name=cmbG_$i  -W  -o slurm_logs/combine_$i.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; source activate ClearMap ; time python -u /gpfs/Labs/Kim/TubeMap_data/scripts/generic_combine_sh_target.py $nodes_to_use_combine_probabilities $i $overlap_x $debug $sixteen_bit_lower_thresh $input_dir1 $input_dir2 $input_file_generic $combine_output_base  $total_files_ch00 $output_tiff_dir $SCRIPT" &
	
	# # i=1
	# # input_file_generic='binary_postprocessed'
	# # combine_output_base='binary_recombined'
	# # output_tiff_dir='binary_filled_recombined'
	# # sbatch -p $target_nodes --cpus-per-task=20 --mem=40G  --job-name=cmbG_$i  -W  -o slurm_logs/combine_$i.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; source activate ClearMap ; time python -u /gpfs/Labs/Kim/TubeMap_data/scripts/generic_combine_sh_target.py $nodes_to_use_combine_probabilities $i $overlap_x $debug $sixteen_bit_lower_thresh $input_dir1 $input_dir2 $input_file_generic $combine_output_base  $total_files_ch00 $output_tiff_dir $SCRIPT" &
	# # i=1
	# # input_file_generic='16bit_probs'
	# # combine_output_base='raw_recombined'
	# # output_tiff_dir='ilastik_binary_recombined'
	# # sbatch -p $target_nodes --cpus-per-task=20 --mem=40G  --job-name=cmbG_$i  -W  -o slurm_logs/combine_$i.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; source activate ClearMap ; time python -u /gpfs/Labs/Kim/TubeMap_data/scripts/generic_combine_sh_target.py $nodes_to_use_combine_probabilities $i $overlap_x $debug $sixteen_bit_lower_thresh $input_dir1 $input_dir2 $input_file_generic $combine_output_base  $total_files_ch00 $output_tiff_dir $SCRIPT" &
	# wait
	# chmod -R 2777 ./tubemap/skeleton_recombined_ch00ch01
	# chmod -R 2777 ./tubemap/filled_recombined_ch00ch01
	# chmod -R 2777 ./tubemap/filled_recombined_ch01
	# chmod -R 2777 ./tubemap/filled_recombined_ch00
	# chmod -R 2777 ./tubemap/skeleton_recombined_ch00
	# chmod -R 2777 ./tubemap/skeleton_recombined_ch01
	# chmod -R 2777 ./slurm_logs/
# fi












# if $tubemap_graph;
	# then
	# i=0
	# input_dir='ilastik_binary_ch00'
	# output_dir='ilastik_binary_ch00_filled'
	# input_file_generic='skeleton_skele'

	# # combine_output_base='skeleton_recombined'
	# # if $combine_skele;
	# # sbatch -p dense --exclusive --requeue --job-name=graph_$i  -W  -o slurm_logs/graph_$i.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; source activate ClearMap ; time  xvfb-run --auto-servernum python -u  /gpfs/Labs/Kim/TubeMap_data/scripts/TubeMap_dan_v3.1_sh_target.py $nodes_to_use_binarize $i $tubemap_chunk_overlap_size $SCRIPT graphing $input_dir $output_dir $overlap_x $sixteen_bit_lower_thresh $filling_thresh $resample_factor_for_filling $debug" &
	# sbatch -p dense --cpus-per-task=20 --mem=600G  --requeue --job-name=graph_$i  -W  -o slurm_logs/graph_$i.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; source activate ClearMap ; time  xvfb-run --auto-servernum python -u  /gpfs/Labs/Kim/TubeMap_data/scripts/TubeMap_dan_v3.1_sh_target.py $nodes_to_use_binarize $i $tubemap_chunk_overlap_size $SCRIPT graphing $input_dir $output_dir $overlap_x $sixteen_bit_lower_thresh $filling_thresh $resample_factor_for_filling $debug" &
	# # sbatch -p dense --requeue --exclusive --job-name=graph_$i  -W  -o slurm_logs/graph_$i.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; source activate ClearMap ; time  xvfb-run -d -a python -u  /gpfs/Labs/Kim/TubeMap_data/scripts/TubeMap_dan_v3.1_sh_target.py $nodes_to_use_binarize $i $tubemap_chunk_overlap_size $SCRIPT graphing ilastik_composite_2d_16bit tm_binarized $overlap_x $sixteen_bit_lower_thresh $filling_thresh $resample_factor_for_filling $debug" &
	# wait
	# chmod -R 2777 ./tubemap/*.gt
	# chmod -R 2777 ./slurm_logs/
# fi


# if $post_hoc_anal
	# then
	# i=0
	# input_dir='ilastik_binary_ch00'
	# output_dir='ilastik_binary_ch00_filled'
	# input_file_generic='skeleton_skele'

	# # combine_output_base='skeleton_recombined'
	# # if $combine_skele;
	# # sbatch -p dense --exclusive --requeue --job-name=graph_$i  -W  -o slurm_logs/graph_$i.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; source activate ClearMap ; time  xvfb-run --auto-servernum python -u  /gpfs/Labs/Kim/TubeMap_data/scripts/TubeMap_dan_v3.1_sh_target.py $nodes_to_use_binarize $i $tubemap_chunk_overlap_size $SCRIPT graphing $input_dir $output_dir $overlap_x $sixteen_bit_lower_thresh $filling_thresh $resample_factor_for_filling $debug" &
	# sbatch -p compute --cpus-per-task=10 --mem=50G  --requeue --job-name=PHA_$i  -W  -o slurm_logs/post_hoc_anal_$i.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; source activate ClearMap ; time  xvfb-run --auto-servernum python -u  /gpfs/Labs/Kim/TubeMap_data/scripts/TubeMap_dan_v3.1_sh_target.py $nodes_to_use_binarize $i $tubemap_chunk_overlap_size $SCRIPT post_hoc_anal $input_dir $output_dir $overlap_x $sixteen_bit_lower_thresh $filling_thresh $resample_factor_for_filling $debug" &
	# # sbatch -p dense --requeue --exclusive --job-name=graph_$i  -W  -o slurm_logs/graph_$i.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; source activate ClearMap ; time  xvfb-run -d -a python -u  /gpfs/Labs/Kim/TubeMap_data/scripts/TubeMap_dan_v3.1_sh_target.py $nodes_to_use_binarize $i $tubemap_chunk_overlap_size $SCRIPT graphing ilastik_composite_2d_16bit tm_binarized $overlap_x $sixteen_bit_lower_thresh $filling_thresh $resample_factor_for_filling $debug" &
	# wait
	# chmod -R 2777 ./tubemap/*.gt
	# chmod -R 2777 ./slurm_logs/
# fi



# # if $combine_skele;
	# # then
		# # for i in `seq 1 $nodes_to_use_combine_probabilities`;
		# # do
			# # # if $mix_nodes
			# # # sbatch -p compute --exclusive --job-name=cmbP_$i  -W  -o slurm_logs/combine_probabilities_$i.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; source activate ClearMap ; time python -u /gpfs/Labs/Kim/TubeMap_data/scripts/probability_combine_sh_target.py $nodes_to_use_combine_probabilities $i $debug $SCRIPT" &
			# # if $mix_nodes;
			# # then
				# # # sbatch -p compute --cpus-per-task=10 --mem=16G  --job-name=cmbP_$i  -W  -o slurm_logs/combine_probabilities_$i.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; source activate ClearMap ; time python -u /gpfs/Labs/Kim/TubeMap_data/scripts/probability_combine_sh_target.py $nodes_to_use_combine_probabilities $i $overlap_x $debug $sixteen_bit_lower_thresh $SCRIPT" &
				# # sbatch -p compute --cpus-per-task=10 --mem=16G  --job-name=cmbS_$i  -W  -o slurm_logs/combine_skele_$i.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; source activate ClearMap ; time python -u /gpfs/Labs/Kim/TubeMap_data/scripts/generic_combine_sh_target.py $nodes_to_use_combine_probabilities $i $overlap_x $debug $sixteen_bit_lower_thresh $input_dir1 $inputdir2 $combine_output_base $input_file_generic $total_files_ch00 $SCRIPT" &
			
			# # else
				# # sbatch -p compute --exclusive --job-name=cmbS_$i  -W  -o slurm_logs/combine_skele_$i.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; source activate ClearMap ; time python -u /gpfs/Labs/Kim/TubeMap_data/scripts/generic_combine_sh_target.py $nodes_to_use_combine_probabilities $i $overlap_x $debug $sixteen_bit_lower_thresh $input_dir1 $inputdir2 $combine_output_base $input_file_generic $total_files_ch00 $SCRIPT" &
			# # fi
			# # sleep 1
		# # done
		# # wait
# # fi










# if $tubemap_smooth;
	# then
		# for i in `seq 1 $nodes_to_use_smoothing`;
		# do
			# if $mix_nodes;
			# then
				# sbatch -p compute  --cpus-per-task=11  --mem=36G --job-name=smth_$i  -W  -o slurm_logs/smooth_$i.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; source activate ClearMap ; time  xvfb-run -d python -u  /gpfs/Labs/Kim/TubeMap_data/scripts/TubeMap_dan_v3.1_sh_target.py $nodes_to_use_smoothing $i $tubemap_chunk_overlap_size $SCRIPT post_proc ilastik_composite 1_smoothed $debug" &
			# else
				# sbatch -p compute --exclusive --job-name=smth_$i  -W  -o slurm_logs/smooth_$i.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; source activate ClearMap ; time  xvfb-run -d python -u  /gpfs/Labs/Kim/TubeMap_data/scripts/TubeMap_dan_v3.1_sh_target.py $nodes_to_use_smoothing $i $tubemap_chunk_overlap_size $SCRIPT post_proc ilastik_composite 1_smoothed $debug" &
			# fi
			# sleep 1
		# done
		# wait
		# chmod -R 2777 ./tubemap/
		# chmod -R 2777 ./slurm_logs/
# fi


# # #Ilastik submissions:
# # for i in `seq 1 $nodes_per_channel_to_use`;
# # do
       # # sbatch -p compute --exclusive --job-name=istk0_c$i  -W  -o slurm_logs/ilastik_ch00_chunk_$i.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; module load ilastik/1.3.3 ; source activate ClearMap ; python /gpfs/Labs/Kim/TubeMap_data/scripts/ilastik_apply_chunk_sh_target.py 0 $i $SCRIPT" &
       # # sleep 2
       # # sbatch -p compute --exclusive --job-name=istk1_c$i  -W  -o slurm_logs/ilastik_ch01_chunk_$i.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; module load ilastik/1.3.3 ; source activate ClearMap ; python /gpfs/Labs/Kim/TubeMap_data/scripts/ilastik_apply_chunk_sh_target.py 1 $i $SCRIPT" & 
       # # sleep 2
# # done
# # wait

# # echo 'Done Ilastik Classification!'

# # # #Un-collapse chunk for normalization. Don't do this. use multinode.
# # # # find ./ilastik_results/stitched_00/ -maxdepth 2 -type f -print0 | xargs -0 mv -t ./ilastik_results/stitched_00
# # # # rm -r -- ./ilastik_results/stitched_00/*/

# # # # find ./ilastik_results/stitched_01/ -maxdepth 2 -type f -print0 | xargs -0 mv -t ./ilastik_results/stitched_01
# # # # rm -r -- ./ilastik_results/stitched_01/*/

# # #Normalize original images using Ilastik label outputs



# # for i in `seq 1 $nodes_per_channel_to_use`;
# # do
       # # sbatch -p compute --exclusive --job-name=inorm0_$i  -W -o slurm_logs/inorm_ch00_c$i.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; source activate ClearMap ; python /gpfs/Labs/Kim/TubeMap_data/scripts/ilastik_normalize_sh_target.py 0 $i $SCRIPT" &
       # # sleep 2
       # # sbatch -p compute --exclusive --job-name=inorm1_$i  -W -o slurm_logs/inorm_ch01_c$i.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; source activate ClearMap ; python /gpfs/Labs/Kim/TubeMap_data/scripts/ilastik_normalize_sh_target.py 1 $i $SCRIPT" &
       # # sleep 2
# # done
# # wait

# # echo 'Done ilastik normalization!'


# # # # Re-distribute the ilastik output into chunks. May have to rename these files in order for Tubemap to understand...
# # # # don't have to re-do this first part if you're running this as a pipeline
# # # # Or just save into these same folders.... Whatever.
# # # # BUT, actually do this because maybe you want to re-distribute the # of nodes youwant to use per channel.

# # let nodes_per_channel_to_use=2
# # total_files_ch00=$(find ./stitched_00_ilastik_normalized -maxdepth 1 -type f -name '*.tif' -printf x | wc -c)
# # total_files_ch01=$(find ./stitched_01_ilastik_normalized -maxdepth 1 -type f -name '*.tif' -printf x | wc -c)


# # echo $total_files_ch00
# # echo $total_files_ch01
# # dir_size=$(bc <<< "scale=0;$total_files_ch00/$nodes_per_channel_to_use+1")
# # echo $dir_size
# # n=$((`find ./stitched_00_ilastik_normalized -maxdepth 1 -type f | wc -l`/$dir_size+1))
# # dir_name="chunk_"
# # echo $n

# # #chunk channel 00 #This is where we need to implement the overlap. Overlap by 10 maybe?
# # for i in `seq 1 $n`;
# # do
	# # next_folder_idx=$(bc <<< "scale=0;$i+1")
	
	# # mkdir -p -m2777 "./stitched_00_ilastik_normalized/$dir_name$i";
	# # find ./stitched_00_ilastik_normalized -maxdepth 1 -type f | sort -V | head -n $dir_size | xargs -i mv "{}" "./stitched_00_ilastik_normalized/$dir_name$i"
	# # #copy 10 files to next folder to account for overlap. this will error on the last folder. lazy. fine for now.
	# # if [ "$nodes_per_channel_to_use" -gt "$i" ];
	# # then
		# # mkdir -p -m2777 "./stitched_00_ilastik_normalized/$dir_name$next_folder_idx";
		# # find ./stitched_00_ilastik_normalized/$dir_name$i -maxdepth 1 -type f | sort -V | tail -20  | xargs -i cp "{}" "./stitched_00_ilastik_normalized/$dir_name$next_folder_idx"
	# # fi
# # done

# # #chunk channel 01
# # for i in `seq 1 $n`;
# # do
	# # next_folder_idx=$(bc <<< "scale=0;$i+1")
	# # mkdir -p -m2777 "./stitched_01_ilastik_normalized/$dir_name$i";
	# # find ./stitched_01_ilastik_normalized -maxdepth 1 -type f | sort -V | head -n $dir_size | xargs -i mv "{}" "./stitched_01_ilastik_normalized/$dir_name$i"
	# # if [ "$nodes_per_channel_to_use" -gt "$i" ];
	# # then
		# # mkdir -p -m2777 "./stitched_01_ilastik_normalized/$dir_name$next_folder_idx";
		# # find ./stitched_01_ilastik_normalized/$dir_name$i -maxdepth 1 -type f | sort -V | tail -20  | xargs -i cp "{}" "./stitched_01_ilastik_normalized/$dir_name$next_folder_idx"
	# # fi
# # done

# # echo 'Done normalization!'


# # #Call Tubemap into respective chunks
# # #x11 forwarding for tubemap		xvfb-run -d ?
# # for i in `seq 1 $nodes_per_channel_to_use`;
# # do
        # # sbatch -p gpu --exclusive -D . --job-name=Tube0_$i  -W -o slurm_logs/Tube_ch00_c$i.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; . activate ClearMap ; xvfb-run -d python -u /gpfs/Labs/Kim/TubeMap_data/scripts/TubeMap_dan_v2_sh_target.py 0 $i $SCRIPT binarize" &
        # # sleep 10
        # # sbatch -p gpu --exclusive  -D . --job-name=Tube1_$i  -W -o slurm_logs/Tube_ch01_c$i.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; . activate ClearMap ; xvfb-run -d python -u /gpfs/Labs/Kim/TubeMap_data/scripts/TubeMap_dan_v2_sh_target.py 1 $i $SCRIPT binarize" &
        # # sleep 10
# # done
# # wait

# # echo 'Done Filling!'


# # mkdir -p -m2777 "./tubemap/ch_00/binary_filled";
# # mkdir -p -m2777 "./tubemap/ch_01/binary_filled";

# # # # ## Un-collapse chunk for skeletonization. No consideration for overlap!!!
# # # # find ./tubemap/ch_00/chunk_*/binary_filled*/ -maxdepth 2 -type f -print0 | xargs -0 mv -t ./tubemap/ch_00/binary_filled
# # # # # rm -r -- ./tubemap/ch_00/chunk_*/binary_filled*/

# # # # find ./tubemap/ch_01/chunk_*/binary_filled*/ -maxdepth 2 -type f -print0 | xargs -0 mv -t ./tubemap/ch_01/binary_filled
# # # # # rm -r -- ./tubemap/ch_01/chunk_*/binary_filled*/

# # #Now consider overlapping!!!
# # #unchunk unoverlap channel 00
# # for i in `seq 1 $n`;
# # do
	# # if [ "$i" -eq "1" ];
	# # then		#tail --lines=+11 head --lines=+11
		# # find ./tubemap/ch_00/chunk_$i/binary_filled*/ -maxdepth 1 -type f  | sort -V  | head --lines=-10  | xargs -i mv "{}" "./tubemap/ch_00/binary_filled"
	# # elif [ "$n" -eq "$i" ];
	# # then
		# # find ./tubemap/ch_00/chunk_$i/binary_filled*/ -maxdepth 1 -type f  | sort -V  | tail --lines=+11  | xargs -i mv "{}" "./tubemap/ch_00/binary_filled"
	# # else
		# # find ./tubemap/ch_00/chunk_$i/binary_filled*/ -maxdepth 1 -type f  | sort -V  | tail --lines=+11  | head --lines=-10 | xargs -i mv "{}" "./tubemap/ch_00/binary_filled"
	# # fi
# # done
# # #unchunk unoverlap channel 01
# # for i in `seq 1 $n`;
# # do
	# # if [ "$i" -eq "1" ];
	# # then
		# # find ./tubemap/ch_01/chunk_$i/binary_filled*/ -maxdepth 1 -type f  | sort -V  | head --lines=-10  | xargs -i mv "{}" "./tubemap/ch_01/binary_filled"
	# # elif [ "$n" -eq "$i" ];
	# # then
		# # find ./tubemap/ch_01/chunk_$i/binary_filled*/ -maxdepth 1 -type f  | sort -V  | tail --lines=+11  | xargs -i mv "{}" "./tubemap/ch_01/binary_filled"
	# # else
		# # find ./tubemap/ch_01/chunk_$i/binary_filled*/ -maxdepth 1 -type f  | sort -V  | tail --lines=+11  | head --lines=-10 | xargs -i mv "{}" "./tubemap/ch_01/binary_filled"
	# # fi
# # done


# # echo 'Restacked and unoverlapped Filled data!'

# # echo 'Registration Initialized!!!'

# # sbatch -p dense --exclusive -D  . --job-name=TubeReg  -W -o slurm_logs/Tube_register.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; . activate ClearMap ; xvfb-run -d python -u /gpfs/Labs/Kim/TubeMap_data/scripts/TubeMap_dan_v2_sh_target.py 0 1 $SCRIPT register" &
# # wait
# # # sbatch -p dense -D . --exclusive --job-name=TubeReg  -W -o slurm_logs/Tube_register.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; . activate ClearMap ; xvfb-run -d python -u /gpfs/Labs/Kim/TubeMap_data/scripts/TubeMap_dan_v2_sh_target.py 0 1 $SCRIPT register" &
# # # wait


# # # combine filled stacks
# # sbatch -p dense --exclusive -D . --job-name=TubeComb  -W -o slurm_logs/Tube_combine.txt --wrap=" . /etc/profile.d/modules.sh ; module load miniconda/3 ; . activate ClearMap ; xvfb-run -d  python -u /gpfs/Labs/Kim/TubeMap_data/scripts/TubeMap_dan_v2_sh_target.py 0 1 $SCRIPT final_binary" &
# # wait
# # echo 'Combined binaries!'

# # # # ## skeletonize both channels (only need binary final though so don't do normally)
# # # # sbatch -p dense -D . --exclusive --job-name=TubeSkel_00  -W -o slurm_logs/Tube_skel_00.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; . activate ClearMap ; xvfb-run -d python -u /gpfs/Labs/Kim/TubeMap_data/scripts/TubeMap_dan_v2_sh_target.py 0 1 $SCRIPT skeletonize" &
# # # # sleep 10
# # # # sbatch -p dense -D . --exclusive --job-name=TubeSkel_01  -W -o slurm_logs/Tube_skel_01.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; . activate ClearMap ; xvfb-run -d python -u /gpfs/Labs/Kim/TubeMap_data/scripts/TubeMap_dan_v2_sh_target.py 1 1 $SCRIPT skeletonize" &
# # # # sleep 10
# # # # wait

# # #Skeletonize the combined binary
# # sbatch -p dense --exclusive -D . --job-name=TubeSkel  -W -o slurm_logs/Tube_skel.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; . activate ClearMap ; xvfb-run -d python -u /gpfs/Labs/Kim/TubeMap_data/scripts/TubeMap_dan_v2_sh_target.py 1 1 $SCRIPT skeletonize" &
# # wait


# # echo 'Skeletonization finished!'


# # #Uncollapse sorta raw data for graph tracing:
# # find ./stitched_00_ilastik_normalized -maxdepth 2 -type f  -print0 | xargs -0 mv -t ./stitched_00_ilastik_normalized/
# # wait
# # rm -r -- ./stitched_00_ilastik_normalized/*/
# # wait
# # find ./stitched_01_ilastik_normalized -maxdepth 2 -type f  -print0 | xargs -0 mv -t ./stitched_01_ilastik_normalized/
# # wait
# # rm -r -- ./stitched_01_ilastik_normalized/*/
# # wait


# # sbatch -p dense --exclusive -D . --job-name=TubeTrace  -W -o slurm_logs/Tube_trace.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; . activate ClearMap ; xvfb-run -d python -u /gpfs/Labs/Kim/TubeMap_data/scripts/TubeMap_dan_v2_sh_target.py 0 1 $SCRIPT graph_trace" &
# # wait

# # # sbatch -p dense -D . --exclusive --job-name=ArtTime  -W -o slurm_logs/Tube_traceArt.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; . activate ClearMap ; xvfb-run -d python -u /gpfs/Labs/Kim/TubeMap_data/scripts/TubeMap_dan_v2_sh_target.py 0 1 $SCRIPT solo_art" &
# # # wait

# # sbatch -p dense --exclusive -D . --job-name=TubeVal  -W -o slurm_logs/Tube_traceArt.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; . activate ClearMap ; xvfb-run -d python -u /gpfs/Labs/Kim/TubeMap_data/scripts/TubeMap_dan_v2_sh_target.py 0 1 $SCRIPT validate" &
# # wait

# # sbatch -p gpu --exclusive -D . --job-name=TubeAnal  -W -o slurm_logs/Tube_anal.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; . activate ClearMap ; xvfb-run -d python -u /gpfs/Labs/Kim/TubeMap_data/scripts/TubeMap_dan_v2_sh_target.py 0 1 $SCRIPT analysis" &
# # wait





# # #Validation 1k x 1k x 1k



# # # # # ## Graph trace both channels?!
# # # # sbatch -p dense -D . --exclusive --job-name=TubeTrace_00  -W -o slurm_logs/Tube_trace_00.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; . activate ClearMap ; xvfb-run -d python -u /gpfs/Labs/Kim/TubeMap_data/scripts/TubeMap_dan_v2_sh_target.py 0 1 $SCRIPT graph_trace" &
# # # # sleep 10
# # # # sbatch -p dense -D . --exclusive --job-name=TubeTrace_01  -W -o slurm_logs/Tube_trace_01.txt --wrap=". /etc/profile.d/modules.sh ; module load miniconda/3 ; . activate ClearMap ; xvfb-run -d python -u /gpfs/Labs/Kim/TubeMap_data/scripts/TubeMap_dan_v2_sh_target.py 1 1 $SCRIPT graph_trace" &
# # # # sleep 10
# # # # wait


# # ##don't do this
# # # find ./stitched_00 -maxdepth 2 -type f  -print0 | xargs -0 mv -t ./stitched_00/
# # # find ./stitched_01 -maxdepth 2 -type f  -print0 | xargs -0 mv -t ./stitched_01/
# # # rm -r -- ./stitched_00/*/
# # # rm -r -- ./stitched_01/*/
# # # rm -r ./ilastik_results/
# # # rm -r ./tubemap/
# # # rm -r ./stitched*ilastik_normalized
# # # rm -r ./slurm_logs
# # # echo 'things are reset try not to fuck up again k?'
# # # echo 'things are reset try not to fuck up again k?'

# # find ./ -type f -or -type d -exec chmod 2777 {} \;

# # if $chmod_stuff;
	# # chmod -R 2777 ./
	# # chmod -R 2777 ../slurm_logs/
	# # wait
# # fi
