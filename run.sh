### Removing models ###
rm_model(){
    local task_name="$1"
    if [[ -z "$task_name" ]]; then
        echo "Usage: rm_model <task-name>"
        return 1
    fi

    echo "Removing task: $task_name"
    rm logs/experiment_test/model_dir/model_${task_name}_seed_1/*
    rm logs/experiment_test/buffer/buffer/buffer_${task_name}_seed_1/*
    rm logs/experiment_test/buffer/buffer_distill/buffer_distill_${task_name}_seed_1/0_*
    rm logs/experiment_test/buffer/buffer_distill_tmp/buffer_distill_tmp_${task_name}_seed_1/*
}
rm_student(){
    local task_name="$1"
    if [[ -z "$task_name" ]]; then
        echo "Usage: rm_model <task-name>"
        return 1
    fi

    echo "Removing student: $task_name"
    rm logs/experiment_test/model_dir/student_model_${task_name}_seed_1/*
}

rm_col_model(){
    echo "Removing col_model!"
    rm logs/experiment_test/model_dir/model_col/*
}


### Training models ###
train_task(){
    local task_name="$1"
    local nr_steps="$2"
    shift 2  # Remove the first two arguments from the list

    if [[ -z "$task_name" || -z "$nr_steps" ]]; then
        echo "Usage: train_task <task-name> <nr-steps> [additional args]"
        return 1
    fi

    echo "Training expert on task: $task_name with $nr_steps steps"
    echo "Additional args: $@"

    python3 -u main.py \
        setup=metaworld \
        env=metaworld-mt1 \
        worker.multitask.num_envs=1 \
        experiment.mode=train_worker \
        env.benchmark.env_name="${task_name}" \
        experiment.num_train_steps="${nr_steps}" \
        "$@"
}

online_distill(){
    local task_name="$1"

    if [[ -z "$task_name" ]]; then
        echo "Usage: $0 <task-name>"
        return 1
    fi

    echo "Distill online: $task_name"

    python3 -u main.py \
        setup=metaworld \
        env=metaworld-mt1 \
        worker.multitask.num_envs=1 \
        experiment.mode=online_distill_collective_transformer \
        env.benchmark.env_name="${task_name}"
}

train_student(){
    local task_name="$1"
    local nr_steps="$2"
    shift 2  # Remove the first two arguments from the list

    if [[ -z "$task_name" || -z "$nr_steps" ]]; then
        echo "Usage: train_student <task-name> <nr-steps> [additional args]"
        return 1
    fi

    echo "Training student on task: $task_name with $nr_steps steps"
    echo "Additional args: $@"

    python3 -u main.py \
        setup=metaworld \
        env=metaworld-mt1 \
        worker.multitask.num_envs=1 \
        experiment.mode=train_student \
        env.benchmark.env_name="${task_name}" \
        experiment.num_student_online_trainsteps2="${nr_steps}" \
        experiment.expert_train_step="${nr_steps}"
        "$@"
        #experiment.num_train_steps="${nr_steps}" \
}

### Evaluating models ###
evaluate_task() {
    local task_name="$1"
    if [[ -z "$task_name" ]]; then
        echo "Usage: evaluate_task <task-name>"
        return 1
    fi

    echo "Evaluating expert for task: $task_name"

    rm ${PROJECT_ROOT}/logs/experiment_test/evaluation_models/*
    cp ${PROJECT_ROOT}/logs/experiment_test/model_dir/model_${task_name}_seed_1/* ${PROJECT_ROOT}/logs/experiment_test/evaluation_models/

    local result_path="${PROJECT_ROOT}/logs/results/worker/$task_name"

    python3 -u main.py \
        setup=metaworld \
        env=metaworld-mt1 \
        worker.multitask.num_envs=1 \
        experiment.mode=evaluate_collective_transformer \
        env.benchmark.env_name=${task_name} \
        experiment.evaluate_transformer="agent" | tee -a $result_path
}

evaluate_student() {
    local task_name="$1"
    if [[ -z "$task_name" ]]; then
        echo "Usage: evaluate_task <task-name>"
        return 1
    fi

    echo "Evaluating student for task: $task_name"

    rm ${PROJECT_ROOT}/logs/experiment_test/evaluation_models/*
    cp ${PROJECT_ROOT}/logs/experiment_test/model_dir/student_model_${task_name}_seed_1/* ${PROJECT_ROOT}/logs/experiment_test/evaluation_models/

    local result_path="${PROJECT_ROOT}/logs/results/student/$task_name"
    python3 -u main.py \
        setup=metaworld \
        env=metaworld-mt1 \
        worker.multitask.num_envs=1 \
        experiment.mode=evaluate_collective_transformer \
        env.benchmark.env_name=${task_name} \
        experiment.evaluate_transformer="agent" | tee -a $result_path
}

evaluate_col_agent(){
    local task_name="$1"
    if [[ -z "$task_name" ]]; then
        echo "Usage: $0 <task-name>"
        return 1
    fi

    echo "Evaluating col_network for task: $task_name"
    local result_path="${PROJECT_ROOT}/logs/results/col/$task_name"
    python3 -u main.py \
        setup=metaworld \
        env=metaworld-mt1 \
        worker.multitask.num_envs=1 \
        experiment.mode=evaluate_collective_transformer \
        env.benchmark.env_name="$task_name" \
        experiment.evaluate_transformer="collective_network"  | tee -a $result_path
}


### Utils ###
split_buffer(){
    local task_name="$1"
    if [[ -z "$task_name" ]]; then
        echo "Usage: $0 <task-name>"
        return 1
    fi

    echo "Evaluating task: $task_name"
    python split_buffer_files.py \
        --source ${PROJECT_ROOT}/logs/experiment_test/buffer/buffer_distill/buffer_distill_${task_name}_seed_1 \
        --train  ${PROJECT_ROOT}/Transformer_RNN/dataset/train/buffer_distill_${task_name}_seed_1 \
        --val    ${PROJECT_ROOT}/Transformer_RNN/dataset/validation/buffer_distill_${task_name}_seed_1
}


split_online_buffer(){
    local task_name="$1"
    if [[ -z "$task_name" ]]; then
        echo "Usage: $0 <task-name>"
        return 1
    fi

    echo "Evaluating task: $task_name"

    python split_buffer_files.py \
        --source logs/experiment_test/buffer/online_buffer_${task_name} \
        --train  logs/experiment_test/buffer/collective_buffer/train/online_buffer_${task_name}_seed_1 \
        --val    logs/experiment_test/buffer/collective_buffer/validation/online_buffer_${task_name}_seed_1
}


print_results(){
    echo
    echo "RESULTS expert:"
    grep -r Evaluation ${PROJECT_ROOT}/logs/results/worker | uniq | grep Evaluation
    echo "RESULTS collective network:"
    grep -r Evaluation ${PROJECT_ROOT}/logs/results/col | uniq | grep Evaluation
    echo "RESULTS student:"
    grep -r Evaluation ${PROJECT_ROOT}/logs/results/student | uniq | grep Evaluation
    echo
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  #echo "Script was executed"
    echo "Executing..."
else
  #echo "Script was sourced"
    return
fi


#############################################################################################
#       execution                                                                           #
#############################################################################################

mkdir -p ${PROJECT_ROOT}/logs/results/worker
mkdir -p ${PROJECT_ROOT}/logs/results/col
mkdir -p ${PROJECT_ROOT}/logs/results/student

# train experts
train_task reach-v2 100000 worker.builder.actor_update_freq=1
train_task push-v2 900000
train_task pick-place-v2 2400000
train_task door-open-v2 1000000
train_task drawer-open-v2 500000  # maybe more samples
train_task drawer-close-v2 200000
train_task button-press-topdown-v2 500000 # may need some finetuning
train_task peg-insert-side-v2 1300000
train_task window-open-v2 300000
train_task window-close-v2 400000 # weird solution


# evaluate single agents
evaluate_task reach-v2 # -> 96/100
evaluate_task push-v2
evaluate_task pick-place-v2
evaluate_task door-open-v2 # -> 100/100
evaluate_task drawer-open-v2
evaluate_task drawer-close-v2 # -> 100/100
evaluate_task button-press-topdown-v2
evaluate_task peg-insert-side-v2
evaluate_task window-open-v2 # -> 98/100
evaluate_task window-close-v2


# prepare dataset for col network
online_distill reach-v2
online_distill push-v2
online_distill pick-place-v2
online_distill door-open-v2
online_distill drawer-open-v2
online_distill drawer-close-v2
online_distill button-press-topdown-v2
online_distill peg-insert-side-v2
online_distill window-open-v2
online_distill window-close-v2

# split and mv dataset for trajectory transformer training
split_buffer reach-v2
split_buffer push-v2
split_buffer pick-place-v2
split_buffer door-open-v2
split_buffer drawer-open-v2
split_buffer drawer-close-v2
split_buffer button-press-topdown-v2
split_buffer peg-insert-side-v2
split_buffer window-open-v2
split_buffer window-close-v2

# split and mv dataset for col network training
split_online_buffer reach-v2
split_online_buffer push-v2
split_online_buffer pick-place-v2
split_online_buffer door-open-v2
split_online_buffer drawer-open-v2
split_online_buffer drawer-close-v2
split_online_buffer button-press-topdown-v2
split_online_buffer peg-insert-side-v2
split_online_buffer window-open-v2
split_online_buffer window-close-v2



# train trajectoryTransformer
python3 Transformer_RNN/dataset_tf.py

mv Transformer_RNN/decision_tf_dataset/_chunk_0 Transformer_RNN/decision_tf_dataset/train/_chunk_0
mv Transformer_RNN/decision_tf_dataset/_chunk_1 Transformer_RNN/decision_tf_dataset/validation/_chunk_0

python3 Transformer_RNN/RepresentationTransformerWithCLS.py

# train col network
python3 -u main.py setup=metaworld env=metaworld-mt1 worker.multitask.num_envs=1 experiment.mode=distill_collective_transformer 






evaluate_col_agent reach-v2
evaluate_col_agent push-v2
evaluate_col_agent pick-place-v2
evaluate_col_agent door-open-v2
evaluate_col_agent drawer-open-v2
evaluate_col_agent drawer-close-v2
evaluate_col_agent button-press-topdown-v2
evaluate_col_agent peg-insert-side-v2
evaluate_col_agent window-open-v2
evaluate_col_agent window-close-v2



train_student reach-v2 100000
train_student push-v2 100000
train_student pick-place-v2 100000
train_student door-open-v2 100000
train_student drawer-open-v2 100000
train_student drawer-close-v2 100000
train_student button-press-topdown-v2 100000
train_student peg-insert-side-v2 100000
train_student window-open-v2 100000
train_student window-close-v2 100000


evaluate_student reach-v2
evaluate_student push-v2
evaluate_student pick-place-v2
evaluate_student door-open-v2
evaluate_student drawer-open-v2
evaluate_student drawer-close-v2
evaluate_student button-press-topdown-v2
evaluate_student peg-insert-side-v2
evaluate_student window-open-v2
evaluate_student window-close-v2


print_results
