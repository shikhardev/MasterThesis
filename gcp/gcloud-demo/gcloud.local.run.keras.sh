gcloud ai-platform jobs submit training CIFAR10_NADAM \
--module-name=trainer.task \
--package-path=./trainer \
--job-dir=gs://data_shikhar \
--region=us-central1 \
--config=trainer/cloudml-gpu.yaml
