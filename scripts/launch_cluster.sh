#!/bin/bash

# Takes three arguments:
#   bucket name - one that has already been created
#   name of key file - without .pem extension
#   number of slave instances
#      ex. bash launch_cluster.sh mybucket mypem 2
# >>./launch_cluster.sh s3-aka-emr-cluster emr_cluster 4
# This script assumes that the file bootstrap-emr.sh is
#   in your current directory.

# Requires the awscli to be set up, need to have correct default region configured
# Run `aws configure` to set this up

aws s3 cp bootstrap-emr.sh s3://$1/scripts/bootstrap-emr.sh

aws emr create-cluster \
    --name PySparkCluster \
    --release-label emr-5.13.0 \
    --applications Name=Spark Name=Hadoop \
    --ec2-attributes KeyName=$2 \
    --use-default-roles \
    --instance-groups \
      InstanceGroupType=MASTER,InstanceCount=1,InstanceType=m5.4xlarge \
      InstanceGroupType=CORE,InstanceCount=$3,InstanceType=m5.4xlarge \
    --bootstrap-actions Path=s3://$1/scripts/bootstrap-emr.sh \
    --log-uri s3://$1/
