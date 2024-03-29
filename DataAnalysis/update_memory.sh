#!/bin/bash

# Find all jobs on hold from user lsperi
jobs_on_hold=$(condor_q -const 'JobStatus == 5 && Owner == "lsperi"' -format "%d\n" ClusterId)

# Loop through each job ID
for job_id in $jobs_on_hold; do
    echo "Updating memory request for job $job_id..."

    # Find the job's submit description file (SDF)
    condor_q -format "%s\n" Args -const "JobID == $job_id" > job.sdf

    # Modify the request_memory parameter in the SDF
    sed -i 's/^request_memory = .*/request_memory = 32.0GB/' job.sdf

    # Resubmit the job using the modified SDF
    condor_submit job.sdf

    echo "Job $job_id memory request updated."
done
