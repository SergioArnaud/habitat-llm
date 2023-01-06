# Variables
output_folder="new_policies"
tag="train"
tasks="nav_to_obj pick place open_cab close_cab open_fridge close_fridge"

# Name of the run
run="${adaptation_strategy}_${tag}"

for task in $tasks; do
    echo "Running $task"
    run_name="${task}_${tag}"

    folder="${output_folder}/${tag}/${task}"
    # make dir if it doesnt exist
    mkdir -p $folder
    mkdir -p $folder/checkpoints

    # Slurm outputs
    output_filename="${folder}/slurm.out"
    output_eval_filename="${folder}/slurm.eval.out"
    error_filename="${folder}/slurm.err"
    error_eval_filename="${folder}/slurm.eval.err"

    # Job name
    job_name="${run_name}_train"
    eval_job_name="${run_name}_eval"

    # Send train job
    #train_job_id=$(sbatch --parsable --output $output_filename --job-name $job_name --error $error_filename  train_rl_skill.sh $task $run_name $folder)
    #echo "Train job id: $train_job_id"

    eval=$(sbatch --parsable --output $output_eval_filename --job-name $job_name --error $error_eval_filename eval_rl_skill.sh $task $run_name $folder)
    echo "eval job id: $eval"

    # Send eval job (start after train starts)
    #eval_job_id=$(sbatch --dependency=after:$train_job_id --parsable --output $output_eval_filename --job-name $eval_job_name --error $error_eval_filename --mem-per-cpu $eval_cpu eval.sh $task $sensor $backbone $run_name $eval_environments)
    #echo "Eval job id: $eval_job_id after $train_job_id with $eval_environments environments and $eval_cpu cpu"

    echo "----------------------------------------"
done

