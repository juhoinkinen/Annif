.PHONY: cld_deploy
cld_deploy: validate-template
	aws cloudformation deploy --template-file annif_stack.yaml --stack-name AnnifStack --capabilities CAPABILITY_IAM

.PHONY: validate-template
validate-template:
	aws cloudformation validate-template --template-body file://annif_stack.yaml

.PHONY: delete_stack
delete_stack:
	#aws ecr delete-repository --repository-name test-repository --force
	aws cloudformation delete-stack --stack-name AnnifStack
	#aws s3 rb s3://juhos-cloudformation-test-bucket

