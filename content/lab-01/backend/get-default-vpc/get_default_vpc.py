import json
import boto3
import cfnresponse
from logging import logger
import traceback

ec2 = boto3.client('ec2')

def lambda_handler(event, context):     
    try:         
        if 'RequestType' in event and event['RequestType'] == 'Create':
            vpc_id = get_default_vpc_id()
            subnets =  get_subnets_for_vpc(vpc_id)
            cfnresponse.send(event, context, cfnresponse.SUCCESS, {'VpcId': vpc_id , "Subnets" : subnets}, '')
        else:
            cfnresponse.send(event, context, cfnresponse.SUCCESS, {},'')
    except:
        logger.exception(f"CFGetDefaultVpcIdTut:failed :{traceback.format_exc()}")
        cfnresponse.send(event, context, cfnresponse.FAILED, {})

def get_default_vpc_id():
    vpcs = ec2.describe_vpcs(Filters=[{'Name': 'is-default', 'Values': ['true']}])
    vpcs = vpcs['Vpcs']
    vpc_id = vpcs[0]['VpcId']
    return vpc_id

def get_subnets_for_vpc(vpcId):
    response = ec2.describe_subnets(
        Filters=[
            {
                'Name': 'vpc-id',
                'Values': [vpcId]
            }
        ]
    )
    subnet_ids = []
    for subnet in response['Subnets']:
        subnet_ids.append(subnet['SubnetId'])
    return subnet_ids