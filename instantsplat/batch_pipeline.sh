#!/bin/bash     # Tells the system to run this script using the bash interpreter
set -euo pipefail     # -e: Exit the script if any command fails, -u: Throw an error for unbound variables, -o pipefail: Normally, a pipeline (cmd1 | cmd2) only fails if the last command fails. This makes it fail if any command in the chain fails.

# Batch version of pipeline.sh
# Traverse DATAROOT/{SAMPLE_TYPE}/index_{SAMPLE_ID}/video.mp4
# and run the same reconstruction steps for each sample.

# No spaces are allowed arouund "=" in bash scripts
# $ {VAR:-default_value}: Parameter expansion syntax. Use the value of VAR if it's set, otherwise use default_value.
DATAROOT="${DATAROOT:-/workspace/DimensionX/data/dimensionx_batch_sat360}"
NUM_FRAMES="${NUM_FRAMES:-35}"
MAX_ITER="${MAX_ITER:-30000}"
LAMBDA_LPIPS="${LAMBDA_LPIPS:-0.3}"
USE_CONFIDENCE=true
CUDA_DEVICE="${CUDA_DEVICE:-0}"
CHECKPOINT_ITERS="${CHECKPOINT_ITERS:-30000}"
SAVE_ITERS="${SAVE_ITERS:-30000}"

# Optional filters (comma-separated):
# SAMPLE_TYPES="photorealistic,stylized"
# SAMPLE_IDS="0001,0042, etc..."
SAMPLE_TYPES="${SAMPLE_TYPES:-}"
SAMPLE_IDS="${SAMPLE_IDS:-}"

SKIP_EXISTING=true
SKIP_EXISTING_MODE="${SKIP_EXISTING_MODE:-files}"   # files|folders
# Stop the batch as soon as a sample fails (default). Use --no_fail_fast to continue.
FAIL_FAST=true

# This is how Bash parses a list of flags: By reading the first argument, popping it off the list, and matching it against the case statement.
while [[ "$#" -gt 0 ]]; do      # While number of arguments passed into script ("$#") is greater that 0
    case "$1" in      
        --dataroot) DATAROOT="$2"; shift 2 ;;
        --num_frames) NUM_FRAMES="$2"; shift 2 ;;
        --iter) MAX_ITER="$2"; shift 2 ;;
        --lambda_lpips) LAMBDA_LPIPS="$2"; shift 2 ;;
        --cuda) CUDA_DEVICE="$2"; shift 2 ;;
        --sample_types) SAMPLE_TYPES="$2"; shift 2 ;;
        --sample_ids) SAMPLE_IDS="$2"; shift 2 ;;
        --save_iters) SAVE_ITERS="$2"; shift 2 ;;
        --checkpoint_iters) CHECKPOINT_ITERS="$2"; shift 2 ;;
        --no_use_confidence) USE_CONFIDENCE=false; shift ;;
        --no_skip_existing) SKIP_EXISTING=false; shift ;;
        --skip_existing_mode) SKIP_EXISTING_MODE="$2"; shift 2 ;;
        --fail_fast) FAIL_FAST=true; shift ;;          # backwards-compatible
        --no_fail_fast) FAIL_FAST=false; shift ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

if [[ "$SKIP_EXISTING_MODE" != "files" && "$SKIP_EXISTING_MODE" != "folders" ]]; then
    echo "Invalid --skip_existing_mode: ${SKIP_EXISTING_MODE} (expected: files or folders)"
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
# IFS (Internal Field Separator) is a Bash variable that determines how the shell splits words when it is processing a command line.
IFS=',' read -r -a TYPE_FILTER_ARR <<< "$SAMPLE_TYPES"  # read -r: Read raw input (no backslashes), -a: Array assignment
IFS=',' read -r -a ID_FILTER_ARR <<< "$SAMPLE_IDS"  # <<<: A "here-string" to pass a string of input to a command.

# in_filter function: Checks if a value is in an array.
in_filter() {
    local value="$1"    # local: Variables are local to the function, meaning they are not visible outside of the function.
    shift    # shift: Shift the positional parameters to the left, discarding the first argument.
    local arr=("$@")    # $@: All positional parameters are expanded as separate words.
    if [[ ${#arr[@]} -eq 0 ]] || [[ -z "${arr[0]}" ]]; then # If the array is empty or the first element is empty, return success (0)
        return 0    # return 0: Return success (0)
    fi  # End of if statement
    local item
    for item in "${arr[@]}"; do # for item in "${arr[@]}": Loop through each element in the array.
        if [[ "$value" == "$item" ]]; then # If the value is equal to the current element, return success (0)
            return 0
        fi    # End of if statement
    done    # End of for loop
    return 1    # return 1: Return failure (1)
}

# run_pipeline_for_sample function: Runs the pipeline for a single sample.
run_pipeline_for_sample() {
    local sample_type="$1"
    local sample_id="$2"
    local video_path="$3"
    local dataset="${sample_type}/index_${sample_id}"
    local image_path="./data/images/${dataset}"
    local scene_output="data/scenes/${dataset}/output_${MAX_ITER}_lpips_${LAMBDA_LPIPS}"
    local conf_suffix=""
    local final_ply_path=""
    local final_checkpoint_path=""

    if [[ "$USE_CONFIDENCE" == true ]]; then
        conf_suffix="_use_conf"
    fi

    scene_output="${scene_output}${conf_suffix}"
    final_ply_path="${scene_output}/point_cloud/iteration_${MAX_ITER}/point_cloud.ply"
    final_checkpoint_path="${scene_output}/chkpnt${MAX_ITER}.pth"

    echo "=========="
    echo "Processing ${dataset}"
    echo "Video: ${video_path}"

    # Skip mode:
    # - files   (default): skip only if final artifacts exist.
    # - folders: skip if output directory exists (useful when files are moved away but
    #            empty output folders are intentionally left behind).
    if [[ "$SKIP_EXISTING" == true ]]; then
        if [[ "$SKIP_EXISTING_MODE" == "folders" ]]; then
            if [[ -d "$scene_output" ]]; then
                echo "Skip ${dataset}: output folder exists at ${scene_output}"
                return 2
            fi
        else
            if [[ -f "$final_ply_path" || -f "$final_checkpoint_path" ]]; then
                echo "Skip ${dataset}: already completed at ${scene_output}"
                return 2
            fi
        fi
    fi

    python get_frame.py "$video_path" "$image_path" "$NUM_FRAMES"
    if [[ $? -ne 0 ]]; then
        echo "get_frame.py failed for ${dataset}"
        return 1
    fi

    python dust3r_inference.py --dataset "$dataset"
    if [[ $? -ne 0 ]]; then
        echo "dust3r_inference.py failed for ${dataset}"
        return 1
    fi

    local -a save_args=()
    local -a ckpt_args=()
    local -a save_arr=()
    local -a ckpt_arr=()

    if [[ -n "$SAVE_ITERS" ]]; then
        read -r -a save_arr <<< "$SAVE_ITERS"
        save_args=(--save_iterations "${save_arr[@]}")
    fi
    if [[ -n "$CHECKPOINT_ITERS" ]]; then
        read -r -a ckpt_arr <<< "$CHECKPOINT_ITERS"
        ckpt_args=(--checkpoint_iterations "${ckpt_arr[@]}")
    fi

    if [[ "$USE_CONFIDENCE" == true ]]; then
        python 3dgs.py --dataset "$dataset" --iter "$MAX_ITER" --use_confidence --lambda_lpips "$LAMBDA_LPIPS" "${save_args[@]}" "${ckpt_args[@]}"
    else
        python 3dgs.py --dataset "$dataset" --iter "$MAX_ITER" --lambda_lpips "$LAMBDA_LPIPS" "${save_args[@]}" "${ckpt_args[@]}"
    fi
}

if [[ ! -d "$DATAROOT" ]]; then
    echo "DATAROOT does not exist: $DATAROOT"
    exit 1
fi

# If data is under an extra wrapper dir (common after unzip), e.g.
#   DATAROOT/dimensionx_batch_sat360/photorealistic/index_0000/video.mp4
# instead of:
#   DATAROOT/photorealistic/index_0000/video.mp4
# then descend once so SAMPLE_TYPE is the real category folder.
has_type_index_pair=false
for type_dir in "$DATAROOT"/*; do
    [[ -d "$type_dir" ]] || continue
    for index_dir in "$type_dir"/index_*; do
        if [[ -d "$index_dir" ]]; then
            has_type_index_pair=true
            break 2
        fi
    done
done
if [[ "$has_type_index_pair" == false ]]; then
    for wrap in "$DATAROOT"/*; do
        [[ -d "$wrap" ]] || continue
        for type_dir in "$wrap"/*; do
            [[ -d "$type_dir" ]] || continue
            for index_dir in "$type_dir"/index_*; do
                if [[ -d "$index_dir" ]]; then
                    echo "Note: samples are one level deeper; using DATAROOT=${wrap}"
                    DATAROOT="$wrap"
                    break 3
                fi
            done
        done
    done
fi

shopt -s nullglob

total=0
success=0
failed=0
skipped=0

for type_dir in "$DATAROOT"/*; do
    [[ -d "$type_dir" ]] || continue
    sample_type="$(basename "$type_dir")"

    if ! in_filter "$sample_type" "${TYPE_FILTER_ARR[@]}"; then
        continue
    fi

    for index_dir in "$type_dir"/index_*; do
        [[ -d "$index_dir" ]] || continue
        index_name="$(basename "$index_dir")"
        sample_id="${index_name#index_}"
        video_file="${index_dir}/video.mp4"

        if ! in_filter "$sample_id" "${ID_FILTER_ARR[@]}"; then
            continue
        fi

        if [[ ! -f "$video_file" ]]; then
            echo "Skip ${sample_type}/${index_name}: missing video.mp4"
            continue
        fi

        total=$((total + 1))
        # We want to continue the batch on "skipped" (status 2), but still be able to
        # stop on real failures when fail-fast is enabled. Temporarily disable `-e`
        # only around the sample run so we can reliably capture its exit code.
        set +e
        run_pipeline_for_sample "$sample_type" "$sample_id" "$video_file"
        status=$?
        set -e

        if [[ $status -eq 0 ]]; then
            success=$((success + 1))
        elif [[ $status -eq 2 ]]; then
            skipped=$((skipped + 1))
        else
            failed=$((failed + 1))
            if [[ "$FAIL_FAST" == true ]]; then
                echo "Stopping due to failure (fail-fast enabled)."
                echo "Processed: ${total}, Success: ${success}, Skipped: ${skipped}, Failed: ${failed}"
                exit 1
            fi
        fi
    done
done

echo "=========="
echo "Batch finished."
echo "Processed: ${total}, Success: ${success}, Skipped: ${skipped}, Failed: ${failed}"

if [[ $failed -gt 0 ]]; then
    exit 1
fi

